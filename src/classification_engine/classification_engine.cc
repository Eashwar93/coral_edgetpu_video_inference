//
// Created by eashwara on 14.05.20.
//

#include "classification_engine.h"
#include "opencv2/opencv.hpp"

#include <algorithm>
#include <queue>
#include <tuple>


namespace edge {


	void ClassificationEngine::img_overlay(cv::Mat& frame, const std::vector<ClassificationCandidate>& ret)
{
		int y_coordinate = 20;
		for (const auto& i : ret)
		{
			cv::putText(frame, "Classname :"+i.classname, cv::Point(15,y_coordinate), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(255,0,0),1.5);
			cv::putText(frame, "Score :"+std::to_string(i.score), cv::Point(15,y_coordinate+20), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(0,255,0),1.5);
			y_coordinate = y_coordinate+40;
		}

	cv::imshow("Classified output", frame);

}

	std::vector<ClassificationCandidate> ClassificationEngine::ClassifyWithOutputVector(const std::vector<float>& inf_vec,
	                                                                      const float& threshold, const bool& verbose) {
		size_t idx = 0;
		float  Max_score = 0;
		size_t max_idx =0;
		std::vector<ClassificationCandidate> ret;
		ClassificationCandidate max_candidate;

		std::for_each(inf_vec.cbegin(), inf_vec.cend(), [&](const float& score) {

			bool change = false;
			if (score >threshold && score > Max_score) {
				Max_score = score;
				max_idx = idx;
				max_candidate.classname = m_labels.at(max_idx);
				max_candidate.score = Max_score;
				change = true;
			}
			idx++;
			if(change){
				ret.push_back(max_candidate);
			}
			if(ret.size()>3)
			{
				ret.erase(ret.begin(),ret.end()-3);
			}
		});
		std::reverse(ret.begin(),ret.end());
		if (verbose)
		{std::cout<< "Top 3 Classification scores:" << std::endl;

			for(const auto& i : ret){
				std::cout << "Class :" << i.classname <<std::endl;
				std::cout << "Score :" << i.score << std::endl;
				std::cout<< "=====================================" << std::endl;
		}
		}
		return ret;
	}
}
