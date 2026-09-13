/*
 * Copyright (c) 2024 ZephyrMLForge
 * SPDX-License-Identifier: Apache-2.0
 *
 * Output handler implementation for TFLite Micro inference.
 * Outputs structured metrics for the pipeline to parse.
 */

#include "output_handler.hpp"

#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

void ReportMetrics(uint64_t average_ns, size_t arena_used, int num_runs, float last_output)
{
	/* Latency is formatted from integer nanoseconds rather than a float divide:
	 * a sub-microsecond inference truncates to 0 in the us field, and the host
	 * needs a figure that survives that. */
	uint32_t latency_us = static_cast<uint32_t>(average_ns / 1000U);
	uint32_t latency_ms_whole = static_cast<uint32_t>(average_ns / 1000000U);
	uint32_t latency_ms_fraction = static_cast<uint32_t>(average_ns % 1000000U);

	printk("METRICS_START\n");
	printk("latency_ns=%llu\n", average_ns);
	printk("latency_us=%u\n", latency_us);
	printk("latency_ms=%u.%06u\n", latency_ms_whole, latency_ms_fraction);
	printk("arena_used=%zu\n", arena_used);
	printk("num_runs=%d\n", num_runs);
	printk("output_val=%.6f\n", static_cast<double>(last_output));
	printk("METRICS_END\n");
	printk("Inference complete\n");
}

void ReportInference(const float *values, int count)
{
	printk("OUT=");
	for (int i = 0; i < count; i++) {
		printk("%s%.6f", i == 0 ? "" : ",", static_cast<double>(values[i]));
	}
	printk("\n");
}
