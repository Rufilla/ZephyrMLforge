/*
 * Copyright (c) 2024 ZephyrMLForge
 * SPDX-License-Identifier: Apache-2.0
 *
 *  Classifies 100 Fashion-MNIST test images held in flash & reports how
 * many it got right.
 */

#include "inference.h"
#include "model.hpp"
#include "test_images.hpp"

#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

namespace {

constexpr size_t kTensorArenaBytes = 32 * 1024;

alignas(16) uint8_t tensor_arena[kTensorArenaBytes];

}  /* namespace */

int main(void)
{
	/*  The USB serial port re-enumerates after the reset that follows
	 * flashing; output sent before this is lost. */
	k_msleep(1000);
	printk("\n=== Fashion-MNIST test: %d images ===\n", kTestImageCount);

	if (!mlforge::init(g_model, tensor_arena, kTensorArenaBytes)) {
		return 0;
	}

	if (mlforge::input_count() != kImagePixels || mlforge::output_count() != kClassCount) {
		printk("error: model takes %d inputs & gives %d outputs, expected %d & %d\n",
		       mlforge::input_count(), mlforge::output_count(), kImagePixels, kClassCount);
		return 0;
	}

	int correct = 0;
	uint32_t latency_sum_us = 0;

	for (int i = 0; i < kTestImageCount; i++) {
		/*  Scaled to [0, 1], as the images were in training. */
		for (int pixel = 0; pixel < kImagePixels; pixel++) {
			mlforge::set_input(pixel, g_test_images[i][pixel] / 255.0f);
		}

		uint32_t elapsed_us = 0;

		if (!mlforge::invoke(&elapsed_us)) {
			printk("error: inference %d failed\n", i);
			return 0;
		}

		const int predicted = mlforge::predicted_class();
		const int label = g_test_labels[i];

		correct += predicted == label;
		latency_sum_us += elapsed_us;

		printk("%3d  label=%-10s  predicted=%-10s  %s\n", i + 1, g_class_names[label],
		       g_class_names[predicted], predicted == label ? "ok" : "WRONG");
	}

	printk("\nimages=%d correct=%d accuracy=%d%% mean_latency_us=%u\n", kTestImageCount,
	       correct, 100 * correct / kTestImageCount, latency_sum_us / kTestImageCount);
	return 0;
}
