cmake_minimum_required(VERSION 3.11)
project(egdetpu_video_inference)
find_package( OpenCV REQUIRED)

set(CMAKE_C_FLAGS "-Wall -pthread")
set(CMAKE_C_FLAGS_DEBUG "-g -O0")
set(CMAKE_C_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -std=c++11 -lstdc++")
set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})

# GET CPU Architechture
if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
    message("Building for x86_64 platform.")
    set(ARCH "k8")
    set(TF_COMMAND make -j8 BUILD_WITH_NNAPI=false -C ${CMAKE_BINARY_DIR}/tensorflow/src/tensorflow -f tensorflow/lite/tools/make/Makefile lib)
    set(TF_INSTALL_PREFIX "linux_x86_64")
elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
    message("Building for aarch64 platform.")
    set(ARCH "${CMAKE_SYSTEM_PROCESSOR}")
    set(TF_INSTALL_PREFIX "generic-aarch64_armv8-a")
    set(CROSS_PREFIX "aarch64-linux-gnu-")
    set(TF_COMMAND make TARGET=generic-aarch64 TARGET_ARCH=armv8-a -C ${CMAKE_BINARY_DIR}/tensorflow/src/tensorflow -f tensorflow/lite/tools/make/Makefile CC=${CROSS_PREFIX}g++ CXX=${CROSS_PREFIX}g++ AR=${CROSS_PREFIX}ar CFLAGS=-fpermissive lib)
elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "armv7l")
    message("Building for armv7l platform using the rpi script from tensorflow.")
    set(ARCH "${CMAKE_SYSTEM_PROCESSOR}")
    set(TF_INSTALL_PREFIX "rpi_armv7l")
    set(CROSS_PREFIX "arm-linux-gnueabihf-")
    set(TF_COMMAND make -j4 TARGET=rpi TARGET_ARCH=armv7l -C ${CMAKE_BINARY_DIR}/tensorflow/src/tensorflow -f tensorflow/lite/tools/make/Makefile CC=${CROSS_PREFIX}g++ CXX=${CROSS_PREFIX}g++ AR=${CROSS_PREFIX}ar CFLAGS=-fpermissive lib)
else()
    message(FATAL_ERROR "Not implemented to build for: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

message("System ARCH is ${CMAKE_SYSTEM_PROCESSOR} ${ARCH}")

include(ExternalProject)


# Must match with commit that libedgetpu.so was built from.
set(TENSORFLOW_COMMIT d855adfc5a0195788bf5f92c3c7352e638aa1109)
ExternalProject_Add(tensorflow
    GIT_REPOSITORY https://github.com/tensorflow/tensorflow
    GIT_TAG ${TENSORFLOW_COMMIT}
    PREFIX "tensorflow"
    CONFIGURE_COMMAND ./tensorflow/lite/tools/make/download_dependencies.sh
    BUILD_IN_SOURCE 1
    BUILD_COMMAND ${TF_COMMAND}
    BUILD_BYPRODUCTS libtensorflow-lite.a ${CMAKE_BINARY_DIR}/tensorflow/src/tensorflow/tensorflow/lite/tools/make/downloads/fft2d/fftsg.c
    BUILD_BYPRODUCTS libtensorflow-lite.a ${CMAKE_BINARY_DIR}/tensorflow/src/tensorflow/tensorflow/lite/tools/optimize/sparsity/format_converter.cc
    INSTALL_COMMAND cp -f ${CMAKE_BINARY_DIR}/tensorflow/src/tensorflow/tensorflow/lite/tools/make/gen/${TF_INSTALL_PREFIX}/lib/libtensorflow-lite.a ${CMAKE_BINARY_DIR}/
)
# Setting Lib path
set(TF_LITE_LIB "${CMAKE_BINARY_DIR}/libtensorflow-lite.a")
set(LIB_EDGETPU "${CMAKE_SOURCE_DIR}/libedgetpu/direct/${ARCH}/libedgetpu.so.1.0")

# Setting all include and link directories.
##########################################################################################################################
include_directories(${CMAKE_SOURCE_DIR}/)
include_directories(${CMAKE_SOURCE_DIR}/src/)
##########################################################################################################################
# project source
include_directories(${CMAKE_SOURCE_DIR}/src/utils)
include_directories(${CMAKE_SOURCE_DIR}/src/common_engine)
include_directories(${CMAKE_SOURCE_DIR}/src/classification_engine)
include_directories(${CMAKE_SOURCE_DIR}/src/image_preprocessing)
include_directories(${CMAKE_SOURCE_DIR}/src/detection_engine)
include_directories(${CMAKE_SOURCE_DIR}/src/humanpose_engine)

##########################################################################################################################
include_directories(${CMAKE_SOURCE_DIR}/libedgetpu/)
include_directories(${CMAKE_SOURCE_DIR}/include)
##########################################################################################################################
# project includes
include_directories(${CMAKE_SOURCE_DIR}/include/utils)
include_directories(${CMAKE_SOURCE_DIR}/include/common_engine)
include_directories(${CMAKE_SOURCE_DIR}/include/classification_engine)
include_directories(${CMAKE_SOURCE_DIR}/include/image_preprocessing)
include_directories(${CMAKE_SOURCE_DIR}/include/detection_engine)
include_directories(${CMAKE_SOURCE_DIR}/include/humanpose_engine)

##########################################################################################################################
include_directories(${CMAKE_SOURCE_DIR}/include/thirdparty/cxxopts)
include_directories(${CMAKE_BINARY_DIR}/tensorflow/src/tensorflow/)
include_directories(${CMAKE_BINARY_DIR}/tensorflow/src/tensorflow/tensorflow/lite/tools/make/downloads/absl/)
include_directories(${CMAKE_BINARY_DIR}/tensorflow/src/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include)
include_directories(${CMAKE_BINARY_DIR}/glog_install/include)
##########################################################################################################################
link_directories(${CMAKE_BINARY_DIR})
##########################################################################################################################
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/${ARCH})

add_library(label_utils
  src/utils/label_utils.cc
  include/utils/label_utils.h)

add_library(pose_decoder
        src/humanpose_engine/posenet_decoder_op.cc
        src/humanpose_engine/posenet_decoder.cc
        include/humanpose_engine/posenet_decoder.h
        include/humanpose_engine/posenet_decoder_op.h)
target_link_libraries(pose_decoder ${TF_LITE_LIB})
add_dependencies(pose_decoder tensorflow)

add_library(engine
        src/common_engine/engine.cc
        include/common_engine/engine.h)
target_link_libraries(engine label_utils pose_decoder ${TF_LITE_LIB})
add_dependencies(engine label_utils pose_decoder tensorflow)

add_library(classification_engine
        src/classification_engine/classification_engine.cc
        include/classification_engine/classification_engine.h)
target_link_libraries(classification_engine engine pose_decoder ${TF_LITE_LIB} ${OpenCV_LIBS})
add_dependencies(classification_engine engine pose_decoder)

add_library(image_preprocessing
        src/image_preprocessing/img_prep.cc
        include/image_preprocessing/img_prep.h)
target_link_libraries(image_preprocessing ${OpenCV_LIBS})

add_library(detection_engine
        src/detection_engine/detection_engine.cc
        include/detection_engine/detection_engine.h)
target_link_libraries(detection_engine engine pose_decoder ${TF_LITE_LIB} ${OpenCV_LIBS})
add_dependencies(detection_engine engine )

add_library(humanpose_engine
        src/humanpose_engine/humanpose_engine.cc
        include/humanpose_engine/humanpose_engine.h)
target_link_libraries(humanpose_engine engine pose_decoder ${TF_LITE_LIB} ${OpenCV_LIBS})
add_dependencies(humanpose_engine engine pose_decoder)

add_executable(classification_camera
        src/classification_camera.cc
        ${CMAKE_BINARY_DIR}/tensorflow/src/tensorflow/tensorflow/lite/tools/make/downloads/fft2d/fftsg.c
        ${CMAKE_BINARY_DIR}/tensorflow/src/tensorflow/tensorflow/lite/tools/optimize/sparsity/format_converter.cc
        )
target_link_libraries(classification_camera image_preprocessing classification_engine engine label_utils ${OpenCV_LIBS} ${TF_LITE_LIB} ${LIB_EDGETPU})
add_dependencies(classification_camera image_preprocessing classification_engine engine label_utils )

add_executable(detection_camera
        src/detection_camera.cc
        ${CMAKE_BINARY_DIR}/tensorflow/src/tensorflow/tensorflow/lite/tools/make/downloads/fft2d/fftsg.c
        ${CMAKE_BINARY_DIR}/tensorflow/src/tensorflow/tensorflow/lite/tools/optimize/sparsity/format_converter.cc
        )
target_link_libraries(detection_camera image_preprocessing detection_engine engine label_utils ${OpenCV_LIBS} ${TF_LITE_LIB} ${LIB_EDGETPU})
add_dependencies(detection_camera image_preprocessing detection_engine engine label_utils )

add_executable(humanpose_camera
        src/humanpose_camera.cc
       ${CMAKE_BINARY_DIR}/tensorflow/src/tensorflow/tensorflow/lite/tools/make/downloads/fft2d/fftsg.c
        ${CMAKE_BINARY_DIR}/tensorflow/src/tensorflow/tensorflow/lite/tools/optimize/sparsity/format_converter.cc
        )
target_link_libraries(humanpose_camera image_preprocessing humanpose_engine engine label_utils pose_decoder ${OpenCV_LIBS} ${TF_LITE_LIB} ${LIB_EDGETPU})
add_dependencies(humanpose_camera image_preprocessing humanpose_engine engine label_utils pose_decoder tensorflow)



