//
// Created by eashwara on 18.05.20.
// A common engine class on top of which other specific purpose engines are built
//

#ifndef EGDETPU_VIDEO_INFERENCE_ENGINE_H
#define EGDETPU_VIDEO_INFERENCE_ENGINE_H

#include <array>
#include <map>
#include <string>

#include "edgetpu.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"

namespace edge {
	class Engine {
	public:
		//Constructors to slightly modify the engine types
		Engine(const std::string& model_path, const std::string& label_path,
						const std::shared_ptr<edgetpu::EdgeTpuContext>& edgetpu_context, bool edgetpu);
		Engine(const std::string& model_path,const std::shared_ptr<edgetpu::EdgeTpuContext>& edgetpu_context,
						bool edgetpu);
		//Prepares the Engine
		void PrepEngine(const std::string& model_path,const std::shared_ptr<edgetpu::EdgeTpuContext>& edgetpu_context, bool edgetpu);
		// Initializes a tflite::Interpreter for CPU usage.
		void InitTfLiteWrapper();
		// Initializes a tflite::Interpreter with edgetpu custom ops.
		void InitTfLiteWrapperEdgetpu(const std::shared_ptr<edgetpu::EdgeTpuContext>& edgetpu_context);
		// Exposes the input tensor shape.
		std::vector<int> GetInputShape();
		// Does Inference with the model and returns the output tensor concatenated as a vector.
		std::vector<float> RunInference(const std::vector<uint8_t>& input_data);



	private:
		std::unique_ptr<tflite::FlatBufferModel> m_model;
		std::unique_ptr<tflite::Interpreter> m_interpreter;
		std::vector<int> m_input_shape;
	public:
		std::map<int, std::string> m_labels;
		std::vector<size_t> m_output_shape;

	};
}

#endif //EGDETPU_VIDEO_INFERENCE_ENGINE_H
