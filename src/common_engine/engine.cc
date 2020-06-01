//
// Created by eashwara on 14.05.20.
//

#include "engine.h"
#include <iostream>
#include <memory>
#include <vector>

#include "label_utils.h"
#include "posenet_decoder_op.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/kernels/register.h"

namespace edge {
	Engine::Engine(
					const std::string& model_path, const std::string& label_path, const std::shared_ptr<edgetpu::EdgeTpuContext>& edgetpu_context,
					const bool edgetpu){
		PrepEngine(model_path,edgetpu_context,edgetpu);
		m_labels = ParseLabel(label_path);
	}
	Engine::Engine(const std::string& model_path,	const std::shared_ptr<edgetpu::EdgeTpuContext>& edgetpu_context,
					const bool edgetpu){
		PrepEngine(model_path,edgetpu_context,edgetpu);
	}
	void Engine::PrepEngine(const std::string &model_path, const std::shared_ptr<edgetpu::EdgeTpuContext> &edgetpu_context,
	                        bool edgetpu) {
		// Loads the model file in the program
		m_model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
		// Initializes interpreter.
		if (edgetpu && edgetpu_context) {
			InitTfLiteWrapperEdgetpu(edgetpu_context);
		}
		else {
			InitTfLiteWrapper();
		}
		m_interpreter->SetNumThreads(1);
		m_interpreter->AllocateTensors();
		// Set input tensor shape.
		const auto* dims = m_interpreter->tensor(m_interpreter->inputs()[0])->dims;
		m_input_shape = {dims->data[0],dims->data[1], dims->data[2], dims->data[3]};
		// set output tensor shape.
		const auto& out_tensor_indices = m_interpreter->outputs();
		m_output_shape.resize(out_tensor_indices.size());
		//for debugging
		//std::cout<<"out_tensor_indices.size() : "<<out_tensor_indices.size()<<std::endl;
		for (size_t i = 0; i < out_tensor_indices.size(); i++) {
			const auto* tensor = m_interpreter->tensor(out_tensor_indices[i]);
			// We are assuming that outputs tensor are only of type float.
			m_output_shape[i] = tensor->bytes / sizeof(float);
		}

	}

	void Engine::InitTfLiteWrapperEdgetpu(
					const std::shared_ptr<edgetpu::EdgeTpuContext>& edgetpu_context) {
		tflite::ops::builtin::BuiltinOpResolver resolver;
		resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
		resolver.AddCustom(coral::kPosenetDecoderOp, coral::RegisterPosenetDecoderOp());
		if (tflite::InterpreterBuilder(*m_model, resolver)(&m_interpreter) != kTfLiteOk) {
			std::cout << "Failed to build Interpreter\n";
			std::abort();
		}
		m_interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context.get());
	}

	void Engine::InitTfLiteWrapper() {
		tflite::ops::builtin::BuiltinOpResolver resolver;
		resolver.AddCustom(coral::kPosenetDecoderOp, coral::RegisterPosenetDecoderOp());
		if (tflite::InterpreterBuilder(*m_model, resolver)(&m_interpreter) != kTfLiteOk) {
			std::cout << "Failed to build Interpreter\n";
			std::abort();
		}
	}

	std::vector<int> Engine::GetInputShape() {
		return m_input_shape;
	}

	std::vector<float> Engine::RunInference (const std::vector<uint8_t>& input_data) {
		std::vector<float> output_data;
		auto* input = m_interpreter->typed_input_tensor<uint8_t>(0);
		std::memcpy(input, input_data.data(), input_data.size());
		m_interpreter->Invoke();

		const auto& output_indices = m_interpreter->outputs();
		const int num_outputs = output_indices.size();
		int out_idx = 0;
		for (int i = 0; i < num_outputs; ++i) {
			const auto* out_tensor = m_interpreter->tensor(output_indices[i]);
			assert(out_tensor != nullptr);
			if (out_tensor->type == kTfLiteUInt8) {
				const int num_values = out_tensor->bytes;
				output_data.resize(out_idx + num_values);
				const uint8_t* output = m_interpreter->typed_output_tensor<uint8_t>(i);
				for (int j = 0; j < num_values; ++j) {
					output_data[out_idx++] =
									(output[j] - out_tensor->params.zero_point) * out_tensor->params.scale;
				}
			} else if (out_tensor->type == kTfLiteFloat32) {
				const int num_values = out_tensor->bytes / sizeof(float);
				output_data.resize(out_idx + num_values);
				const float* output = m_interpreter->typed_output_tensor<float>(i);
				for (int j = 0; j < num_values; ++j) {
					output_data[out_idx++] = output[j];
				}
			} else {
				std::cerr << "Tensor " << out_tensor->name
				          << " has unsupported output type: " << out_tensor->type << std::endl;
			}
		}
		return output_data;
	}
}