/*
 * Copyright (c) 2024 ZephyrMLForge
 * SPDX-License-Identifier: Apache-2.0
 *
 * Minimal main entry point for TFLite Micro inference.
 * Based on TensorFlow Lite Micro hello_world sample structure.
 */

#include "app_config.h"
#include "main_functions.h"

int main(void)
{
	setup();
	run_benchmark();

#if MLFORGE_INTERACTIVE
	while (1) {
		serve_request();
	}
#endif

	return 0;
}
