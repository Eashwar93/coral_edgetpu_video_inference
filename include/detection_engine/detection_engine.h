//
// Created by eashwara on 19.05.20.
//

#ifndef EGDETPU_VIDEO_INFERENCE_DETECTION_ENGINE_H
#define EGDETPU_VIDEO_INFERENCE_DETECTION_ENGINE_H

#include <array>
#include <map>
#include <string>

#include "engine.h"
#include "opencv2/opencv.hpp"

namespace edge {
	//Data structure to hold Detection result
	struct DetectionCandidate {
		std::string candidate;
		float score;
		float x1;
		float y1;
		float x2;
		float y2;
	};

	class DetectionEngine : public Engine {
	public:
		//Constructor that loads the model and label into the program.
		DetectionEngine(const std::string& model, const std::string& label_path,
		                const std::shared_ptr<edgetpu::EdgeTpuContext>& edgetpu_context,
		                const bool edgetpu)
						: Engine(model,label_path,edgetpu_context,edgetpu){
			std::cout << "Detection Engine loaded successfully" << std::endl;

		}
		//Overlay the detection output on the image.
		static void img_overlay(cv::Mat& frame, const std::vector<DetectionCandidate>& ret, const int& width, const int& height);

		//Returns a vector of Detection candidates.
		std::vector<DetectionCandidate> DetectWithOutputVector(
						const std::vector<float>& inf_vec,const float& threshold);

	};
}
#endif //EGDETPU_VIDEO_INFERENCE_DETECTION_ENGINE_H
