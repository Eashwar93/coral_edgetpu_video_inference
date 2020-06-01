//
// Created by eashwara on 23.05.20.
//

#include "humanpose_engine.h"
#include "opencv2/opencv.hpp"

namespace edge {
	void HumanPoseEngine::img_overlay(cv::Mat& frame, const std::vector<PoseCandidate>& ret, const float& keypoint_threshold,
					const float& inp_width, const float& inp_height, const float& camera_width, const float& camera_height)
	{
		std::vector<int> k_x(17), k_y(17);
		const auto& green = cv::Scalar(0,255,0);
		for (auto& candidate : ret)
		{
			for (int i=0; i< 17; i++)
			{
				if(candidate.keypoint_scores[i] > keypoint_threshold)
				{
					float x_coordinate = candidate.keypoint_coordinates[(2*i)+1]*(camera_width/inp_width);
					float y_coordinate = candidate.keypoint_coordinates[2*i]*(camera_height/inp_height);
					k_x[i]= static_cast<int>(x_coordinate);
					k_y[i]= static_cast<int>(y_coordinate);
					cv::circle(frame, cv::Point(k_x[i],k_y[i]), 0, green, 6, 1, 0);
				}
			}
		}
		cv::imshow("Pose Estimation", frame);
	}

	std::vector<PoseCandidate> HumanPoseEngine::PoseEstimateWithOutputVector(const std::vector<float>& inf_vec,
	                                                                        const float& threshold)
	{
		const auto* result_raw = inf_vec.data();
		std::vector<std::vector<float>> results(m_output_shape.size());
		int offset = 0;
		for(size_t i=0; i < m_output_shape.size();++i) {
			const size_t size_of_output_tensor_i = m_output_shape[i];
			results[i].resize(size_of_output_tensor_i);
			std::memcpy(results[i].data(), result_raw + offset, sizeof(float) * size_of_output_tensor_i);
			offset += size_of_output_tensor_i;
		}
		std::vector<PoseCandidate> inf_results;
		int n = lround(results[3][0]);
		for (int i = 0; i < n; i++) {
			float overall_score = results[2][i];
			if(overall_score>threshold) {
				PoseCandidate result;
				std::copy(results[1].begin()+(17*i), results[1].begin()+(17*i)+16, std::back_inserter(result.keypoint_scores));
				std::copy(results[0].begin()+(17*2*i), results[0].begin()+(17*2*i)+33, std::back_inserter((result.keypoint_coordinates)));
				inf_results.push_back(result);
			}
		}
		return inf_results;
	}
}

