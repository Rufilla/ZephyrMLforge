/*
 * Copyright (c) 2024 ZephyrMLForge
 * SPDX-License-Identifier: Apache-2.0
 *
 *  Accelerometer windows & motor control for the ML Vibro Sens Click.
 */

#ifndef ZEPHYR_ML_FORGE_VIBRATION_H_
#define ZEPHYR_ML_FORGE_VIBRATION_H_

#include "app_config.h"

#include <stdint.h>

/*  Three axes per sample, interleaved x, y, z, matching the host's
 * (samples, 1, 3) feature layout once flattened. */
#define MLFORGE_VIBRATION_AXES 3
#define MLFORGE_VIBRATION_VALUES (MLFORGE_WINDOW_SAMPLES * MLFORGE_VIBRATION_AXES)

/**
 * @brief Configure the accelerometer & the motor pins, both motors off.
 * @return 0, or a negative errno with the reason printed as SENSOR=error.
 */
int vibration_init(void);

/**
 * @brief Switch the motors; the two never run together.
 * @param state  0 both off, 1 balanced only, 2 unbalanced only.
 * @return false, with the motors unchanged, for any other state.
 */
bool vibration_set_motors(int state);

/**
 * @brief Record one window at MLFORGE_SAMPLE_RATE_HZ.
 * @param[out] window      Filled with MLFORGE_VIBRATION_VALUES readings in m/s^2,
 *                         interleaved x, y, z, each axis with its window mean removed.
 * @param[out] elapsed_us  Duration of the capture.
 * @return 0, -ETIME when a read overran its sample period, or the sensor's negative
 *         errno.
 */
int vibration_capture(float *window, uint32_t *elapsed_us);

#endif /* ZEPHYR_ML_FORGE_VIBRATION_H_ */
