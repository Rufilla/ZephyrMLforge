/*
 * Copyright (c) 2024 ZephyrMLForge
 * SPDX-License-Identifier: Apache-2.0
 *
 * TFLite Micro set-up.Quantization
 * of inputs & outputs is handled here, so callers work in floats.
 */

#ifndef ZEPHYR_ML_FORGE_APPS_INFERENCE_H_
#define ZEPHYR_ML_FORGE_APPS_INFERENCE_H_

#include <stddef.h>
#include <stdint.h>

namespace mlforge {

/**
 * @brief Initialize the TFLite Micro interpreter with a model & tensor arena.
 * @param model_data  The .tflite flatbuffer; must outlive the interpreter.
 * @param arena       Working memory for the interpreter.
 * @param arena_size  Bytes in arena.
 * @return true, or false with the reason printed when the model's schema version
 *         differs from the interpreter's or the arena is too small.
 */
bool init(const unsigned char *model_data, uint8_t *arena, size_t arena_size);

/**
 * @brief Elements in one input sample, every dimension but a leading batch of one.
 * @return The element count; init() must have succeeded.
 */
int input_count();

/**
 * @brief Elements in one output sample, every dimension but a leading batch of one.
 * @return The element count; init() must have succeeded.
 */
int output_count();

/**
 * @brief Write one input element, quantizing it when the input tensor is int8.
 * @param index  Element below input_count().
 * @param value  Value in the model's input units; int8 inputs saturate at -128 & 127.
 */
void set_input(int index, float value);

/**
 * @brief Read one output element, dequantized when the output tensor is int8.
 * @param index  Element below output_count().
 * @return The element's value.
 */
float output(int index);

/**
 * @brief The index of the largest output, the first on a tie.
 * @return A class index below output_count().
 */
int predicted_class();

/**
 * @brief Run the model on the inputs set so far.
 * @param[out] elapsed_us  Duration of the inference alone.
 * @return false on an Invoke() error.
 */
bool invoke(uint32_t *elapsed_us);

}  /* namespace mlforge */

#endif /* ZEPHYR_ML_FORGE_APPS_INFERENCE_H_ */
