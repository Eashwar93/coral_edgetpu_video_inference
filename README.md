# coral_edgetpu_video_inference

## Dependencies:
1. [OpenCV](https://linuxize.com/post/how-to-install-opencv-on-ubuntu-18-04/)
2. [Upgrade GCC](https://www.youtube.com/watch?v=vVzshfYSgRk) - Upgrade your gcc and g++ to the latest version. Don't worry they are always backward compatible.

## Build:
1. Clone the Repository
```bash
git clone https://github.com/Eashwar93/coral_edgetpu_video_inference.git
cd coral_edgetpu_video_inference/scripts
```
2. Upgrade your cmake to the latest version and again don't worry they are backward compatible.
```bash
bash install_cmake.sh
```
3. Build the repostitory
```bash
cd ..
mkdir build && cd build
cmake ..
make
```
