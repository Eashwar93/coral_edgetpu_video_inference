//
// Created by eashwara on 23.05.20.
//

#ifndef EGDETPU_VIDEO_INFERENCE_HUMANPOSE_ENGINE_H
#define EGDETPU_VIDEO_INFERENCE_HUMANPOSE_ENGINE_H
#include <array>
#include <map>
#include <string>

#include "engine.h"
#include "opencv2/opencv.hpp"

namespace edge{
	//Data structure to hold Detection result
	struct PoseCandidate {
		std::vector<float> keypoint_scores;
		std::vector<float>keypoint_coordinates;

	};

	class HumanPoseEngine : public Engine {
	public:

		//Constructor that loads the model into the program
		HumanPoseEngine(const std::string& model, const std::shared_ptr<edgetpu::EdgeTpuContext>& edgetpu_context,
		               const bool edgetpu)
		               : Engine(model,edgetpu_context,edgetpu){
			std::cout << "Pose Engine loaded successfully" << std::endl;
		}

		//Overlay the pose estimate on the image
		static void img_overlay(cv::Mat& frame, const std::vector<PoseCandidate>& ret,const float& keypoint_threshold,
		                        const float& inp_width, const float& inp_height, const float& camera_width, const float& camera_height);

		//Returns a vector of Pose candidates.
		std::vector<PoseCandidate> PoseEstimateWithOutputVector(
						const std::vector<float>& inf_vec, const float& threshold);

	};
}
#endif //EGDETPU_VIDEO_INFERENCE_HUMANPOSE_ENGINE_H
