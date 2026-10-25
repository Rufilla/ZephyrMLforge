/*
 * Copyright (c) 2024 ZephyrMLForge
 * SPDX-License-Identifier: Apache-2.0
 *
 * Main function declarations for TFLite Micro inference.
 * Based on TensorFlow Lite Micro hello_world sample structure.
 */

#ifndef ZEPHYR_ML_FORGE_MAIN_FUNCTIONS_H_
#define ZEPHYR_ML_FORGE_MAIN_FUNCTIONS_H_

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize the interpreter & print 'STATUS=ready' or 'STATUS=error'.
 *
 * Every later entry point is a no-op once setup has failed.
 */
void setup(void);

/**
 * @brief Time MLFORGE_BENCHMARK_RUNS inferences & print one METRICS block.
 */
void run_benchmark(void);

/**
 * @brief Read one command line from the console & answer it.
 *
 * Blocks until a line arrives. Commands: PING, BENCH, INFER, LOAD & INVOKE, plus
 * MOTOR, WINDOW & CLASSIFY in a vibration build.
 */
void serve_request(void);

#ifdef __cplusplus
}
#endif

#endif /* ZEPHYR_ML_FORGE_MAIN_FUNCTIONS_H_ */
