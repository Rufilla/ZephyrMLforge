/*
 * Copyright (c) 2024 ZephyrMLForge
 * SPDX-License-Identifier: Apache-2.0
 */

#include "inference.h"

#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

#include <cmath>

#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/system_setup.h>
#include <tensorflow/lite/schema/schema_generated.h>

namespace mlforge {
namespace {

tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output_tensor = nullptr;

/**
 * @brief Elements in one sample of tensor, every dimension but the leading batch.
 * @param tensor  An input or output tensor of the initialized interpreter.
 * @return The element count; a one-dimensional tensor counts all its elements.
 */
int element_count(const TfLiteTensor *tensor)
{
	int count = 1;

	for (int i = 1; i < tensor->dims->size; i++) {
		count *= tensor->dims->data[i];
	}
	return tensor->dims->size == 1 ? tensor->dims->data[0] : count;
}

}  /* namespace */

bool init(const unsigned char *model_data, uint8_t *arena, size_t arena_size)
{
	tflite::InitializeTarget();

	const tflite::Model *model = tflite::GetModel(model_data);

	if (model->version() != TFLITE_SCHEMA_VERSION) {
		printk("error: model schema %u, expected %u\n",
		       static_cast<unsigned int>(model->version()),
		       static_cast<unsigned int>(TFLITE_SCHEMA_VERSION));
		return false;
	}

	/*  The operators the pipeline lets the agent propose, as registered in
	 * zephyr_app/src/main_functions.cpp, so any model it trained runs here. */
	static tflite::MicroMutableOpResolver<24> resolver;
	resolver.AddFullyConnected();
	resolver.AddRelu();
	resolver.AddRelu6();
	resolver.AddSoftmax();
	resolver.AddReshape();
	resolver.AddQuantize();
	resolver.AddDequantize();
	resolver.AddLogistic();
	resolver.AddConv2D();
	resolver.AddDepthwiseConv2D();
	resolver.AddMaxPool2D();
	resolver.AddAveragePool2D();
	resolver.AddMean();
	resolver.AddPad();
	resolver.AddPadV2();
	resolver.AddAdd();
	resolver.AddShape();
	resolver.AddStridedSlice();
	resolver.AddPack();
	resolver.AddSlice();

	static tflite::MicroInterpreter static_interpreter(model, resolver, arena, arena_size);
	interpreter = &static_interpreter;

	if (interpreter->AllocateTensors() != kTfLiteOk) {
		printk("error: the model needs a tensor arena larger than %u bytes\n",
		       static_cast<unsigned int>(arena_size));
		return false;
	}

	input = interpreter->input(0);
	output_tensor = interpreter->output(0);
	return true;
}

int input_count()
{
	return element_count(input);
}

int output_count()
{
	return element_count(output_tensor);
}

void set_input(int index, float value)
{
	if (input->type != kTfLiteInt8) {
		input->data.f[index] = value;
		return;
	}

	/* TFLite's quantization, round(x / scale) + zero_point clamped to
	 * int8. A bare cast would truncate, adding up to a step of error, & is undefined
	 * for values outside -128 to 127. */
	float scaled = roundf(value / input->params.scale + input->params.zero_point);

	scaled = scaled > 127.0f ? 127.0f : (scaled < -128.0f ? -128.0f : scaled);
	input->data.int8[index] = static_cast<int8_t>(scaled);
}

float output(int index)
{
	if (output_tensor->type != kTfLiteInt8) {
		return output_tensor->data.f[index];
	}
	return (output_tensor->data.int8[index] - output_tensor->params.zero_point) *
	       output_tensor->params.scale;
}

int predicted_class()
{
	int best = 0;

	for (int i = 1; i < output_count(); i++) {
		if (output(i) > output(best)) {
			best = i;
		}
	}
	return best;
}

bool invoke(uint32_t *elapsed_us)
{
	const uint32_t start = k_cycle_get_32();
	const TfLiteStatus status = interpreter->Invoke();

	*elapsed_us = k_cyc_to_us_floor32(k_cycle_get_32() - start);
	return status == kTfLiteOk;
}

}  /* namespace mlforge */
