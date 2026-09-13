/*
 * Copyright (c) 2024 ZephyrMLForge
 * SPDX-License-Identifier: Apache-2.0
 *
 * TFLite Micro inference implementation.
 * Based on TensorFlow Lite Micro hello_world sample structure.
 */

#include "app_config.h"
#include "main_functions.h"
#include "model.hpp"
#include "output_handler.hpp"

#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <zephyr/timing/timing.h>

#if MLFORGE_INTERACTIVE
#include <zephyr/console/console.h>
#endif

#include <cmath>
#include <cstdlib>
#include <cstring>

#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/system_setup.h>
#include <tensorflow/lite/schema/schema_generated.h>

#if MLFORGE_NEUTRON
#include <tensorflow/lite/micro/kernels/neutron/neutron.h>
#endif

namespace {

const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;
bool is_ready = false;

alignas(16) uint8_t tensor_arena[MLFORGE_TENSOR_ARENA_SIZE];

/* One INFER reply is built on the stack, so the width is capped. */
constexpr int kMaxReportedOutputs = 16;

/* Constant benchmark input: the value does not affect timing. */
constexpr float kBenchmarkInput = 0.5f;

/* Elements in one sample: every dimension but the leading batch of one. Taking only
 * the trailing dimension reports 1 for an image of (1, 28, 28, 1). */
int element_count(const TfLiteTensor *tensor)
{
	if (tensor->dims->size <= 1) {
		return tensor->dims->size == 1 ? tensor->dims->data[0] : 0;
	}

	int count = 1;

	for (int i = 1; i < tensor->dims->size; i++) {
		count *= tensor->dims->data[i];
	}

	return count;
}

void write_input(int index, float value)
{
	if (input->type == kTfLiteInt8) {
		/* Rounded, not truncated: TFLite quantises to nearest, and truncating
		 * biases every input one step toward zero. Measured against the host
		 * interpreter, that moved a prediction by a whole quantisation step. */
		float scaled = roundf(value / input->params.scale + input->params.zero_point);

		/* Saturate: a host sending outside the calibrated range would
		 * otherwise wrap and produce a plausible but wrong prediction. */
		if (scaled > 127.0f) {
			scaled = 127.0f;
		} else if (scaled < -128.0f) {
			scaled = -128.0f;
		}
		input->data.int8[index] = static_cast<int8_t>(scaled);
	} else {
		input->data.f[index] = value;
	}
}

float read_output(int index)
{
	if (output->type == kTfLiteInt8) {
		return (output->data.int8[index] - output->params.zero_point) *
		       output->params.scale;
	}
	return output->data.f[index];
}

}  /* namespace */

void setup(void)
{
	/* USB CDC re-enumerates after the reset that follows flashing; without this
	 * delay the banner and STATUS line are transmitted into a closed port. */
	k_msleep(500);

	tflite::InitializeTarget();

	printk("\n=== ZephyrMLForge inference ===\n");
	printk("model_bytes=%u\n", g_model_len);

	model = tflite::GetModel(g_model);
	if (model->version() != TFLITE_SCHEMA_VERSION) {
		printk("STATUS=error reason=schema_version got=%u want=%u\n",
		       static_cast<unsigned int>(model->version()),
		       static_cast<unsigned int>(TFLITE_SCHEMA_VERSION));
		return;
	}

	/* Every op the agent is permitted to propose. Adding one here without
	 * widening the prompt in agents/base.py hard-faults the board instead. */
	static tflite::MicroMutableOpResolver<24> resolver;
	resolver.AddFullyConnected();
	resolver.AddRelu();
	resolver.AddRelu6();
	resolver.AddSoftmax();
	resolver.AddReshape();
	resolver.AddQuantize();
	resolver.AddDequantize();
	resolver.AddLogistic();

	/* Convolutional models, for the image datasets. */
	resolver.AddConv2D();
	resolver.AddDepthwiseConv2D();
	resolver.AddMaxPool2D();
	resolver.AddAveragePool2D();
	resolver.AddMean();
	resolver.AddPad();
	resolver.AddAdd();

	/* Emitted by the converter rather than chosen by the agent: a Keras Flatten
	 * becomes a dynamic reshape, which reads its target shape at run time. */
	resolver.AddShape();
	resolver.AddStridedSlice();
	resolver.AddPack();
	resolver.AddSlice();

#if MLFORGE_NEUTRON
	/* neutron_compiler folds the operators the NPU can run into this one custom
	 * operator; whatever it could not map stays on the CPU above. */
	resolver.AddCustom(tflite::GetString_NEUTRON_GRAPH(), tflite::Register_NEUTRON_GRAPH());
	printk("accelerator=neutron\n");
#else
	printk("accelerator=cpu\n");
#endif

	static tflite::MicroInterpreter static_interpreter(
		model, resolver, tensor_arena, MLFORGE_TENSOR_ARENA_SIZE);
	interpreter = &static_interpreter;

	if (interpreter->AllocateTensors() != kTfLiteOk) {
		printk("STATUS=error reason=allocate_tensors arena_size=%d\n",
		       MLFORGE_TENSOR_ARENA_SIZE);
		return;
	}

	input = interpreter->input(0);
	output = interpreter->output(0);

#if MLFORGE_INTERACTIVE
	console_getline_init();
#endif

	is_ready = true;
	printk("input_count=%d output_count=%d\n", element_count(input),
	       element_count(output));

	/* Worth printing: a scale of zero means the converter left the tensor
	 * unquantised, and every prediction then collapses to the zero point. */
	printk("input_quant scale=%.8f zero_point=%d\n",
	       static_cast<double>(input->params.scale), input->params.zero_point);
	printk("output_quant scale=%.8f zero_point=%d\n",
	       static_cast<double>(output->params.scale), output->params.zero_point);

	printk("STATUS=ready\n");
}

void run_benchmark(void)
{
	if (!is_ready) {
		return;
	}

	const int input_count = element_count(input);

	timing_init();
	timing_start();

	/* Timed per run, with the input written outside the window. The input must be
	 * rewritten every pass - the memory planner may overlay its buffer with an
	 * intermediate tensor - but writing 784 quantised values costs more than the
	 * inference itself, so it must not be inside the measurement. Accumulated in
	 * 64 bits, so a long benchmark cannot wrap the 32-bit counter. */
	uint64_t total_cycles = 0;
	uint64_t total_ticks = 0;

	for (int run = 0; run < MLFORGE_BENCHMARK_RUNS; run++) {
		for (int i = 0; i < input_count; i++) {
			write_input(i, kBenchmarkInput);
		}

		uint32_t start_ticks = k_cycle_get_32();
		timing_t start_time = timing_counter_get();

		TfLiteStatus status = interpreter->Invoke();

		timing_t end_time = timing_counter_get();
		total_ticks += k_cycle_get_32() - start_ticks;
		total_cycles += timing_cycles_get(&start_time, &end_time);

		if (status != kTfLiteOk) {
			timing_stop();
			printk("STATUS=error reason=invoke run=%d\n", run);
			return;
		}
	}

	timing_stop();

	uint64_t total_ns = timing_cycles_to_ns(total_cycles);

	/* QEMU's Cortex-M3 implements no DWT cycle counter, so the timing subsystem
	 * reports zero there while the kernel counter still advances. Reporting the
	 * zero would look like a free inference to the host. */
	if (total_ns == 0) {
		total_ns = k_cyc_to_ns_floor64(total_ticks);
	}

	uint64_t average_ns = total_ns / MLFORGE_BENCHMARK_RUNS;

	ReportMetrics(average_ns, interpreter->arena_used_bytes(), MLFORGE_BENCHMARK_RUNS,
		      read_output(0));
}

#if MLFORGE_INTERACTIVE

void serve_request(void)
{
	char *line = console_getline();

	if (line == nullptr) {
		return;
	}

	while (*line == ' ') {
		line++;
	}

	if (strncmp(line, "PING", 4) == 0) {
		printk("PONG\n");
		return;
	}

	/* Re-runs the measurement on demand. The block printed at boot is easy for the
	 * host to miss, because the port is still enumerating when it is sent. */
	if (strncmp(line, "BENCH", 5) == 0) {
		run_benchmark();
		return;
	}

	if (strncmp(line, "INFER", 5) != 0) {
		printk("ERR=unknown_command\n");
		return;
	}

	if (!is_ready) {
		printk("ERR=not_ready\n");
		return;
	}

	const int input_count = element_count(input);
	const char *cursor = line + 5;

	for (int i = 0; i < input_count; i++) {
		char *next = nullptr;
		float value = strtof(cursor, &next);

		if (next == cursor) {
			printk("ERR=expected_inputs=%d\n", input_count);
			return;
		}

		write_input(i, value);
		cursor = next;
		while (*cursor == ',' || *cursor == ' ') {
			cursor++;
		}
	}

	if (interpreter->Invoke() != kTfLiteOk) {
		printk("ERR=invoke\n");
		return;
	}

	const int output_count = element_count(output);
	const int reported = output_count < kMaxReportedOutputs ? output_count
							        : kMaxReportedOutputs;
	float values[kMaxReportedOutputs];

	for (int i = 0; i < reported; i++) {
		values[i] = read_output(i);
	}

	ReportInference(values, reported);
}

#else

void serve_request(void)
{
}

#endif /* MLFORGE_INTERACTIVE */
