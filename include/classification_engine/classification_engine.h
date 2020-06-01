//
// Created by eashwara on 18.05.20.
//

#ifndef EGDETPU_VIDEO_INFERENCE_CLASSIFICATION_ENGINE_H
#define EGDETPU_VIDEO_INFERENCE_CLASSIFICATION_ENGINE_H

#include "engine.h"
#include "opencv2/opencv.hpp"

#include <string>
#include <vector>
#include <iostream>


namespace edge {
	//Data structure to hold classification result
	struct ClassificationCandidate {
	std::string classname;
	float score;
	};
	class ClassificationEngine : public Engine{
		//Constructor that loads the model and label into the program
	public:
		ClassificationEngine(const std::string& model, const std::string& label_path,
		                     std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context,
		                     const bool edgetpu)
		                     : Engine(model,label_path,edgetpu_context,edgetpu) {
			std::cout << "Classification Engine loaded successfully" << std::endl;
		}

		//Overlay the classification output on the image
		static void img_overlay(cv::Mat& frame, const std::vector<ClassificationCandidate>& ret);

		//Returns a 3 member vector of classification candidates with highest scores
		std::vector<ClassificationCandidate> ClassifyWithOutputVector(
						const std::vector<float>& inf_vec,const float& threshold,const bool& verbose);

	};
}

#endif //EGDETPU_VIDEO_INFERENCE_CLASSIFICATION_ENGINE_H

