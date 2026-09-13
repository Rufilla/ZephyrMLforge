/*
 * Copyright (c) 2024 ZephyrMLForge
 * SPDX-License-Identifier: Apache-2.0
 *
 * Build-time tunables for the inference application.
 */

#ifndef ZEPHYR_ML_FORGE_APP_CONFIG_H_
#define ZEPHYR_ML_FORGE_APP_CONFIG_H_

/* The pipeline rewrites this file in its copy of the tree from
 * hardware.tensor_arena_kb and execution_environment.benchmark_runs. The values
 * below are defaults that keep this tree buildable on its own. */

/* Must exceed the arena the model actually needs, or AllocateTensors() fails. */
#define MLFORGE_TENSOR_ARENA_SIZE 8192

/* Inferences averaged for the reported latency. */
#define MLFORGE_BENCHMARK_RUNS 100

/* 1 serves host evaluation requests after the benchmark; 0 halts instead.
 * QEMU builds set 0 because the runner has no channel to the guest console. */
#define MLFORGE_INTERACTIVE 1

/* 1 registers the NeutronGraph operator, so a model compiled by neutron_compiler
 * runs on the NPU. Requires CONFIG_TFLM_NXP_NEUTRON=y and a board that has one. */
#define MLFORGE_NEUTRON 0

#endif /* ZEPHYR_ML_FORGE_APP_CONFIG_H_ */
