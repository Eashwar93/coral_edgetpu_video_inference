//
// Created by eashwara on 14.05.20.
//

#ifndef EDGETPU_VIDEO_INFERENCE_IMG_PREP_H
#define EDGETPU_VIDEO_INFERENCE_IMG_PREP_H

#include "opencv2/opencv.hpp"

namespace edge {

	// Converts a image to a input vector for the model.
	std::vector<uint8_t > GetInputFromImage(cv::Mat input_frame, const int& width,
					const int& height, const int& channels);
}
#endif //EDGETPU_VIDEO_INFERENCE_IMG_PREP_H
