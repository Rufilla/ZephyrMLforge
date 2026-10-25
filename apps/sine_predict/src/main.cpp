/*
 * Copyright (c) 2024 ZephyrMLForge
 * SPDX-License-Identifier: Apache-2.0
 *
 *  100 predictions of sin(x) across the range the model was trained on,
 * each printed against the true value, then the error over all of them.
 */

#include "inference.h"
#include "model.hpp"

#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

#include <cmath>

namespace {

constexpr int kPredictions = 100;

/*  The synthetic training data drew x uniformly from [-pi, pi]. */
constexpr float kLowest = -3.14159265f;
constexpr float kHighest = 3.14159265f;

constexpr size_t kTensorArenaBytes = 8 * 1024;

alignas(16) uint8_t tensor_arena[kTensorArenaBytes];

}  /* namespace */

int main(void)
{
	/*  The USB serial port re-enumerates after the reset that follows
	 * flashing; output sent before this is lost. */
	k_msleep(1000);
	printk("\n=== Sine predictions ===\n");

	if (!mlforge::init(g_model, tensor_arena, kTensorArenaBytes)) {
		return 0;
	}

	float squared_error_sum = 0.0f;
	float worst_error = 0.0f;
	uint32_t latency_sum_us = 0;

	for (int i = 0; i < kPredictions; i++) {
		const float x = kLowest + (kHighest - kLowest) * i / (kPredictions - 1);
		uint32_t elapsed_us = 0;

		mlforge::set_input(0, x);
		if (!mlforge::invoke(&elapsed_us)) {
			printk("error: inference %d failed\n", i);
			return 0;
		}

		const float predicted = mlforge::output(0);
		const float expected = sinf(x);
		const float error = predicted - expected;

		squared_error_sum += error * error;
		worst_error = fabsf(error) > worst_error ? fabsf(error) : worst_error;
		latency_sum_us += elapsed_us;

		printk("%3d  x=%+.4f  predicted=%+.4f  sin(x)=%+.4f  error=%+.4f\n", i + 1,
		       static_cast<double>(x), static_cast<double>(predicted),
		       static_cast<double>(expected), static_cast<double>(error));
	}

	printk("\npredictions=%d mse=%.6f max_abs_error=%.4f mean_latency_us=%u\n",
	       kPredictions, static_cast<double>(squared_error_sum / kPredictions),
	       static_cast<double>(worst_error), latency_sum_us / kPredictions);
	return 0;
}
