/*
 * Copyright (c) 2024 ZephyrMLForge
 * SPDX-License-Identifier: Apache-2.0
 *
 *  Runs a randomly chosen motor on the ML Vibro Sens Click, 20 times, &
 * has the model say from one accelerometer window which motor it was.
 */

#include "inference.h"
#include "model.hpp"
#include "vibration.h"

#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

namespace {

constexpr int kTrials = 20;

/*  The model's classes, in the order of the firmware's MOTOR states. */
constexpr int kOff = 0;
const char *const kClassNames[] = {"off", "balanced", "unbalanced"};

/*  As in capture: a motor runs this long before its window is kept. */
constexpr int kSettleMs = 1000;
constexpr int kRestMs = 500;

constexpr size_t kTensorArenaBytes = 16 * 1024;

alignas(16) uint8_t tensor_arena[kTensorArenaBytes];

float window[MLFORGE_VIBRATION_VALUES];

uint32_t random_state;

/**
 * @brief Advance the xorshift32 generator; fair enough to choose a motor, & needs no
 *        entropy driver.
 * @return The next value of random_state.
 */
uint32_t next_random(void)
{
	random_state ^= random_state << 13;
	random_state ^= random_state >> 17;
	random_state ^= random_state << 5;
	return random_state;
}

/**
 * @brief Seed random_state from the accelerometer's noise floor & the cycle counter, so
 *        each power-up picks a different sequence.
 */
void seed_random(void)
{
	uint32_t elapsed_us = 0;

	random_state = k_cycle_get_32() | 1U;
	if (vibration_capture(window, &elapsed_us) == 0) {
		for (int i = 0; i < MLFORGE_VIBRATION_VALUES; i++) {
			const int32_t noise = static_cast<int32_t>(window[i] * 1000000.0f);

			random_state ^= static_cast<uint32_t>(noise) << (i % 24);
			next_random();
		}
	}
}

}  /* namespace */

int main(void)
{
	/*  The USB serial port re-enumerates after the reset that follows
	 * flashing; output sent before this is lost. */
	k_msleep(1000);
	printk("\n=== Which motor is running? %d trials ===\n", kTrials);

	if (vibration_init() != 0 ||
	    !mlforge::init(g_model, tensor_arena, kTensorArenaBytes)) {
		return 0;
	}

	if (mlforge::input_count() != MLFORGE_VIBRATION_VALUES) {
		printk("error: model takes %d inputs, a window is %d\n", mlforge::input_count(),
		       MLFORGE_VIBRATION_VALUES);
		return 0;
	}

	seed_random();

	int correct = 0;

	for (int trial = 1; trial <= kTrials; trial++) {
		/*  1 the balanced motor, 2 the unbalanced one; never both. */
		const int motor = 1 + static_cast<int>(next_random() % 2);
		uint32_t elapsed_us = 0;

		vibration_set_motors(motor);
		k_msleep(kSettleMs);
		const int ret = vibration_capture(window, &elapsed_us);

		vibration_set_motors(kOff);

		if (ret != 0) {
			printk("error: window %d could not be read (%d)\n", trial, ret);
			return 0;
		}

		for (int i = 0; i < MLFORGE_VIBRATION_VALUES; i++) {
			mlforge::set_input(i, window[i]);
		}
		if (!mlforge::invoke(&elapsed_us)) {
			printk("error: inference %d failed\n", trial);
			return 0;
		}

		const int predicted = mlforge::predicted_class();

		correct += predicted == motor;
		printk("%2d  running=%-10s  predicted=%-10s  score=%.2f  %s\n", trial,
		       kClassNames[motor], kClassNames[predicted],
		       static_cast<double>(mlforge::output(predicted)),
		       predicted == motor ? "ok" : "WRONG");

		k_msleep(kRestMs);
	}

	printk("\ntrials=%d correct=%d\n", kTrials, correct);
	return 0;
}
