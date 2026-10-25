/*
 * Copyright (c) 2024 ZephyrMLForge
 * SPDX-License-Identifier: Apache-2.0
 *
 *  Written by apps/export_models.py.
 */

#ifndef ZEPHYR_ML_FORGE_TEST_IMAGES_HPP_
#define ZEPHYR_ML_FORGE_TEST_IMAGES_HPP_

#include <stdint.h>

constexpr int kTestImageCount = 100;
constexpr int kImagePixels = 784;
constexpr int kClassCount = 10;

/*  Grayscale pixels, 0 to 255, row by row. */
extern const uint8_t g_test_images[kTestImageCount][kImagePixels];
extern const uint8_t g_test_labels[kTestImageCount];
extern const char *const g_class_names[kClassCount];

#endif /* ZEPHYR_ML_FORGE_TEST_IMAGES_HPP_ */
