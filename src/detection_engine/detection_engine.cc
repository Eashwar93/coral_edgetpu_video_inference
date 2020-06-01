//
// Created by eashwara on 19.05.20.
//

#include "detection_engine.h"
#include "opencv2/opencv.hpp"

#include <queue>
#include <tuple>

namespace edge {
	void DetectionEngine::img_overlay(cv::Mat& frame, const std::vector<DetectionCandidate>& ret, const int& width, const int& height)
	{
		for (const auto& candidate : ret) {
			int top = static_cast<int>(candidate.y1 * height + 0.5f);
			int lft = static_cast<int>(candidate.x1 * width + 0.5f);
			int btm = static_cast<int>(candidate.y2 * height + 0.5f);
			int rgt = static_cast<int>(candidate.x2 * width + 0.5f);
			const auto &cvred = cv::Scalar(0, 0, 255);
			const auto &cvblue = cv::Scalar(0, 255, 0);
			const auto &c = "candidate: " + candidate.candidate;
			const auto &s = "score: " + std::to_string(candidate.score);

			cv::rectangle(
							frame, cv::Point(lft, top), cv::Point(rgt, btm), cvblue, 2, 1, 0);
			cv::putText(
							frame, c, cv::Point(lft, top - 25), cv::FONT_HERSHEY_COMPLEX, .8, cvred, 1.5, 8, false);
			cv::putText(
							frame, s, cv::Point(lft, top - 5), cv::FONT_HERSHEY_COMPLEX, .8, cvred, 1.5, 8, false);
		}
		cv::imshow("Detections", frame);
	}

	std::vector<DetectionCandidate> DetectionEngine::DetectWithOutputVector(
					const std::vector<float>& inf_vec,const float& threshold)
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
			std::vector<DetectionCandidate> inf_results;
			int n = lround(results[3][0]);
			for (int i = 0; i < n; i++) {
				int id = lround(results[1][i]);
				float score = results[2][i];
				if (score > threshold) {
					DetectionCandidate result;
					result.candidate = m_labels.at(id);
					result.score = score;
					result.y1 = std::max(static_cast<float>(0.0), results[0][4 * i]);
					result.x1 = std::max(static_cast<float>(0.0), results[0][4 * i + 1]);
					result.y2 = std::min(static_cast<float>(1.0), results[0][4 * i + 2]);
					result.x2 = std::min(static_cast<float>(1.0), results[0][4 * i + 3]);
					inf_results.push_back(result);
				}
			}
		return inf_results;
	}
}