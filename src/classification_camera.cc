//
// Created by eashwara on 18.05.20.
//

#include <algorithm>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>

#include "edgetpu.h"
#include "img_prep.h"
#include "classification_engine.h"
#include "cxxopts.hpp"
#include "opencv2/opencv.hpp"

cxxopts::ParseResult parse_args(int argc, char** argv) {
	cxxopts::Options options("edgetpu_video_inference", "Package to use tflite/edgetpu to persform inference on videostream");

	options.add_options()
					("model_path", "Path to .tflite/.edgetpu model_file", cxxopts::value<std::string>())
					("label_path", "Path to label file.", cxxopts::value<std::string>())
					("video_source", "Video source.", cxxopts::value<int>()->default_value("0"))
					("threshold", "Minimum confidence threshold.", cxxopts::value<float>()->default_value("0.1"))
					("verbose", "To run in verbose mode.", cxxopts::value<bool>()->default_value("false"))
					("edgetpu", "To run with EdgeTPU.", cxxopts::value<bool>()->default_value("false"))
					("height", "Camera image height.", cxxopts::value<int>()->default_value("480"))
					("width", "Camera image width.", cxxopts::value<int>()->default_value("640"))
					("help", "Print Usage");

	const auto& args = options.parse(argc, argv);
	if (args.count("help") || !args.count("model_path") || !args.count("label_path")) {
		std::cerr << options.help() << "\n";
		exit(0);
	}
	return args;
}

int main(int argc, char** argv) {
	const auto &args = parse_args(argc, argv);
	// Building Interpreter.
	const auto &model_path = args["model_path"].as<std::string>();
	const auto &label_path = args["label_path"].as<std::string>();
	const auto threshold = args["threshold"].as<float>();
	const auto with_edgetpu = args["edgetpu"].as<bool>();
	auto image_height = args["height"].as<int>();
	auto image_width = args["width"].as<int>();
	const auto source = args["video_source"].as<int>();
	const bool verbose = args["verbose"].as<bool>();

	std::cout << std::endl << "Model Path : " << model_path << std::endl;
	std::cout << "Pose Threshold : " << label_path << std::endl;
	std::cout << "Classification threshold : " << threshold << std::endl;
	std::cout << "TPU Acceleration : " << std::boolalpha << with_edgetpu << std::endl;
	std::cout << "Camera Height : " << image_height << std::endl;
	std::cout << "Camera Width : " << image_width << std::endl;
	std::cout << "Camera Source : " << source << std::endl;
	std::cout << "Verbose Mode : " << std::boolalpha << verbose << std::endl;

	std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context =
					edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();

	edge::ClassificationEngine engine(model_path,label_path,edgetpu_context,with_edgetpu);
	const auto& required_input_tensor_shape = engine.GetInputShape();


	cv::VideoCapture cam_frame;
	cam_frame.open(source);
	if(!cam_frame.set(cv::CAP_PROP_FRAME_HEIGHT, image_height))
	{
		std::cout<<"Camera opened with default Height" << std::endl;
	}
	if(!cam_frame.set(cv::CAP_PROP_FRAME_WIDTH, image_width))
	{
		std::cout<<"Camera opened with default Width" << std::endl;
	}
	if(!cam_frame.set(cv::CAP_PROP_FPS, 30.0f))
	{
		std::cout<<"Camera opened with default FPS" << std::endl;
	}

	if(!cam_frame.isOpened())
	{
		std::cout << "Error opening camera \n";
	}
	cv::Mat frame;
	cam_frame >> frame;
	std::cout << "Camera is opened with resolution (H x W) set at : " << frame.size << std::endl;
	cv::Size camera_resolution = frame.size();
	int camera_width = camera_resolution.width;
	int camera_height = camera_resolution.height;
	if(required_input_tensor_shape[2] > (camera_width+10) || required_input_tensor_shape[1] > (camera_height+10))
	{
		std::cout << "Camera resolution smaller than Model Input Dimension. You can go for smaller version of the model to improve performance" << std::endl;
	}
	if(required_input_tensor_shape[3] != frame.channels())
	{
		std::cout << "The channels of the video-stream doesnt match the input channel dimension of the model" << std::endl;
		return 0;
	}

	while(true)
	{
		cam_frame >> frame;
		const auto& input = edge::GetInputFromImage(frame,required_input_tensor_shape[2],required_input_tensor_shape[1],required_input_tensor_shape[3]);
		const auto& results = engine.RunInference(input);
		const auto& class_result = engine.ClassifyWithOutputVector(results,threshold,verbose);
		edge::ClassificationEngine::img_overlay(frame,class_result);

		char c=(char)cv::waitKey(25);
		if(c==27)
			break;
	}
}