/*
 * Copyright (c) 2024 ZephyrMLForge
 * SPDX-License-Identifier: Apache-2.0
 *
 *  Accelerometer windows & motor control for the ML Vibro Sens Click.
 * Built only with vibration.overlay, which supplies the mlforge_accel node & the motor
 * pins.
 */

#include "vibration.h"

#if MLFORGE_VIBRATION

#include <zephyr/device.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/drivers/i2c.h>
#include <zephyr/drivers/sensor.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

#include <cerrno>

namespace {

/*  FXLS8974 registers. The Zephyr driver's rate attribute writes
 * SENS_CONFIG3 without entering standby, & only accepts SENSOR_CHAN_ALL, & its init
 * sets +-2 g whatever the devicetree range, so both are set here directly. */
constexpr uint8_t kRegSensConfig1 = 0x15;
constexpr uint8_t kRegSensConfig3 = 0x17;
constexpr uint8_t kActiveMask = 0x01;
constexpr uint8_t kRangeMask = 0x06;
constexpr uint8_t kWakeOdrMask = 0xF0;
constexpr uint8_t kWakeOdr3200Hz = 0x00;

/*  The chip's fastest rate. Polled below it, each read returns the
 * newest conversion rather than one up to a period stale. */
constexpr int kSensorRateHz = 3200;

static_assert(MLFORGE_SAMPLE_RATE_HZ > 0 && MLFORGE_SAMPLE_RATE_HZ <= kSensorRateHz,
	      "MLFORGE_SAMPLE_RATE_HZ must be between 1 and 3200");

constexpr int kRangeG = DT_PROP(DT_NODELABEL(mlforge_accel), range);
constexpr uint8_t kRangeBits = kRangeG == 16 ? 0x06 : kRangeG == 8 ? 0x04 : kRangeG == 4 ? 0x02 : 0x00;

const struct device *const accel = DEVICE_DT_GET(DT_NODELABEL(mlforge_accel));
const struct i2c_dt_spec accel_bus = I2C_DT_SPEC_GET(DT_NODELABEL(mlforge_accel));
const struct gpio_dt_spec balanced_motor =
	GPIO_DT_SPEC_GET(DT_PATH(zephyr_user), balanced_motor_gpios);
const struct gpio_dt_spec unbalanced_motor =
	GPIO_DT_SPEC_GET(DT_PATH(zephyr_user), unbalanced_motor_gpios);

K_TIMER_DEFINE(sample_timer, NULL, NULL);

/**
 * @brief Set the FXLS8974's range & output rate; it is put in standby while they change.
 * @return 0, or the I2C transfer's negative errno.
 */
int configure_sensor(void)
{
	int ret = i2c_reg_update_byte_dt(&accel_bus, kRegSensConfig1, kActiveMask, 0);

	if (ret == 0) {
		ret = i2c_reg_update_byte_dt(&accel_bus, kRegSensConfig1, kRangeMask, kRangeBits);
	}
	if (ret == 0) {
		ret = i2c_reg_update_byte_dt(&accel_bus, kRegSensConfig3, kWakeOdrMask,
					     kWakeOdr3200Hz);
	}
	if (ret == 0) {
		ret = i2c_reg_update_byte_dt(&accel_bus, kRegSensConfig1, kActiveMask,
					     kActiveMask);
	}
	return ret;
}

}  /* namespace */

int vibration_init(void)
{
	if (!device_is_ready(accel)) {
		printk("SENSOR=error reason=accel_not_ready\n");
		return -ENODEV;
	}

	if (!gpio_is_ready_dt(&balanced_motor) || !gpio_is_ready_dt(&unbalanced_motor)) {
		printk("SENSOR=error reason=motor_gpio_not_ready\n");
		return -ENODEV;
	}

	int ret = gpio_pin_configure_dt(&balanced_motor, GPIO_OUTPUT_INACTIVE);

	if (ret == 0) {
		ret = gpio_pin_configure_dt(&unbalanced_motor, GPIO_OUTPUT_INACTIVE);
	}
	if (ret != 0) {
		printk("SENSOR=error reason=motor_gpio_configure err=%d\n", ret);
		return ret;
	}

	ret = configure_sensor();
	if (ret != 0) {
		printk("SENSOR=error reason=accel_configure err=%d\n", ret);
		return ret;
	}

	printk("sensor=fxls8974 sample_rate_hz=%d window_samples=%d range_g=%d\n",
	       MLFORGE_SAMPLE_RATE_HZ, MLFORGE_WINDOW_SAMPLES, kRangeG);
	return 0;
}

bool vibration_set_motors(int state)
{
	if (state < 0 || state > 2) {
		return false;
	}

	gpio_pin_set_dt(&balanced_motor, state == 1);
	gpio_pin_set_dt(&unbalanced_motor, state == 2);
	return true;
}

int vibration_capture(float *window, uint32_t *elapsed_us)
{
	const uint32_t period_us = 1000000U / MLFORGE_SAMPLE_RATE_HZ;
	float sums[MLFORGE_VIBRATION_AXES] = {};
	uint32_t missed_periods = 0;

	k_timer_start(&sample_timer, K_USEC(period_us), K_USEC(period_us));
	const int64_t start_ticks = k_uptime_ticks();

	for (int sample = 0; sample < MLFORGE_WINDOW_SAMPLES; sample++) {
		/*  A count above one means a read overran its period, & the
		 * window no longer has the spacing the model is trained on. */
		const uint32_t periods = k_timer_status_sync(&sample_timer);

		if (sample > 0 && periods > 1) {
			missed_periods += periods - 1;
		}

		struct sensor_value values[MLFORGE_VIBRATION_AXES];
		int ret = sensor_sample_fetch(accel);

		if (ret == 0) {
			ret = sensor_channel_get(accel, SENSOR_CHAN_ACCEL_XYZ, values);
		}
		if (ret != 0) {
			k_timer_stop(&sample_timer);
			return ret;
		}

		for (int axis = 0; axis < MLFORGE_VIBRATION_AXES; axis++) {
			const float value = sensor_value_to_float(&values[axis]);

			window[sample * MLFORGE_VIBRATION_AXES + axis] = value;
			sums[axis] += value;
		}
	}

	*elapsed_us = static_cast<uint32_t>(k_ticks_to_us_floor64(k_uptime_ticks() - start_ticks));
	k_timer_stop(&sample_timer);

	if (missed_periods > 0) {
		return -ETIME;
	}

	/*  Removes gravity & the sensor's offset, which depend on how the
	 * board lies rather than on which motor runs. */
	for (int axis = 0; axis < MLFORGE_VIBRATION_AXES; axis++) {
		const float mean = sums[axis] / MLFORGE_WINDOW_SAMPLES;

		for (int sample = 0; sample < MLFORGE_WINDOW_SAMPLES; sample++) {
			window[sample * MLFORGE_VIBRATION_AXES + axis] -= mean;
		}
	}

	return 0;
}

#endif /* MLFORGE_VIBRATION */
