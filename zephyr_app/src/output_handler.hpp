/*
 * Copyright (c) 2024 ZephyrMLForge
 * SPDX-License-Identifier: Apache-2.0
 *
 * Output handler declarations for TFLite Micro inference.
 * Based on TensorFlow Lite Micro hello_world sample structure.
 */

#ifndef ZEPHYR_ML_FORGE_OUTPUT_HANDLER_HPP_
#define ZEPHYR_ML_FORGE_OUTPUT_HANDLER_HPP_

#include <cstddef>
#include <cstdint>

/*
 * Prints the METRICS block the pipeline parses, followed by 'Inference complete'.
 *
 * @param average_ns Mean time for one Invoke() call, in nanoseconds
 * @param arena_used Bytes of the tensor arena the interpreter claimed
 * @param num_runs Inferences the average was taken over
 * @param last_output First element of the final output tensor, as a sanity value
 */
void ReportMetrics(uint64_t average_ns, size_t arena_used, int num_runs, float last_output);

/*
 * Prints one 'OUT=' line answering an INFER request.
 *
 * @param values Dequantised output tensor elements
 * @param count Number of elements in values
 */
void ReportInference(const float *values, int count);

#endif /* ZEPHYR_ML_FORGE_OUTPUT_HANDLER_HPP_ */
