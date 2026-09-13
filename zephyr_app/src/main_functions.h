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

/* Initialises the interpreter and prints 'STATUS=ready' or 'STATUS=error'.
 * Every later entry point is a no-op once setup has failed. */
void setup(void);

/* Times MLFORGE_BENCHMARK_RUNS inferences and prints one METRICS block. */
void run_benchmark(void);

/* Reads one command line from the console and answers it. Blocks until a line
 * arrives. Commands: 'PING', 'INFER <f>[,<f>...]'. */
void serve_request(void);

#ifdef __cplusplus
}
#endif

#endif /* ZEPHYR_ML_FORGE_MAIN_FUNCTIONS_H_ */
