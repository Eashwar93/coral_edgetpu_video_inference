# coral_edgetpu_video_inference

## Prerequisites:
1. A linux machine, I tested this build on x86_64
2. Front Cam or USB Cam connected to your computer with drivers preinstalled
3. Optional - EdgeTPU for NN acceleration
## Dependencies: 
Links attached for your reference
1. [OpenCV](https://linuxize.com/post/how-to-install-opencv-on-ubuntu-18-04/)
2. [Upgrade GCC](https://www.youtube.com/watch?v=vVzshfYSgRk) - Upgrade your gcc and g++ to the latest version. Don't worry they are always backward compatible.
```bash
$ sudo apt install libgtk2.0-dev
$ sudo apt search libgtk2.0-dev
```
## Build:
1. Clone the Repository
```bash
$ git clone https://github.com/Eashwar93/coral_edgetpu_video_inference.git
$ cd coral_edgetpu_video_inference/scripts
```
2. Upgrade your cmake to the latest version and again don't worry they are backward compatible.
```bash
$ bash install_cmake.sh
```
3. Build the repostitory
```bash
$ cd ..
$ mkdir build && cd build
$ cmake ..
$ make
$ cd ..
```
## Usage:
1. If you have a [Coral USB Accelerator](https://coral.ai/products/accelerator/) you run any of the following scripts else skip to Step 2:
The first script runs classification on a video stream,the second runs Object Detection and the third runs human pose estimation 
```bash
$ bash scripts/classification/classify_edgetpu.sh
$ bash scripts/detection/detect_edgetpu.sh
$ bash scripts/pose_estimation/pose_edgetpu_480x640.sh
```
2. The following scripts can be run on computers without any accelerators:
```bash
$ bash scripts/classification/classify_cpu.sh
$ bash scripts/detection/detect_cpu.sh
$ bash scripts/pose_estimation/pose_cpu_353x481.sh
```
## Preview 
I apprently made use of Coral USB Accelerator and below are the results for your reference.
Click [here](https://github.com/Eashwar93/coral_edgetpu_video_inference/tree/master/gifs) to see the GIF Demos.

## Credits
1. [google-coral/edgetpu](https://github.com/google-coral/edgetpu)
2. [Namburger/edgetpu-detection-camera](https://github.com/Namburger/edgetpu-detection-camera/tree/master) 

## Further Work
1. A medium article on how to use Google TPU's and why do we need them.
2. ROS wrapper to take advantage of the the TPU for Robot Naviagtion.

