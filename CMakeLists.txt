cmake_minimum_required(VERSION 3.0)
project(full-yolov5)
set(CMAKE_CXX_STANDARD 14)

#寻找OpenCV库
FIND_PACKAGE(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS}
                    ${CUDA_INCLUDE_DIRS}
                    /home/robot/tensorrt/TensorRT-8.6.1.6/include
                    /home/robot/my_project/A_study/cuda&tensorrt/practice/tensorrt/integrate-1.18-integrate-full-yolov5/src
                    /home/robot/my_project/A_study/cuda&tensorrt/practice/tensorrt/integrate-1.18-integrate-full-yolov5/src/tensorRT)


add_executable(main src/main.cpp
                    src/tensorRT/common/json.cpp
                    src/tensorRT/common/trt_tensor.cpp
#                    src/tensorRT/common/preprocess_kernel.cu
                    src/tensorRT/common/cuda_tools.cpp
                    src/tensorRT/common/ilogger.cpp
                    src/tensorRT/builder/trt_builder.cpp
                    src/tensorRT/infer/trt_infer.cpp
                    src/app_yolo/yolo.cpp
#                    src/app_yolo/yolo_decode.cu
                    )

cuda_add_library(preprocess_kernel src/tensorRT/common/preprocess_kernel.cu)
cuda_add_library(yolo_decode src/app_yolo/yolo_decode.cu)
target_link_libraries(main preprocess_kernel
                           yolo_decode)

# 链接.so文件
target_link_libraries(main /home/robot/tensorrt/TensorRT-8.6.1.6/lib/libnvinfer.so
                           /usr/local/cuda-11.3/lib64/libcudart.so
                           /home/robot/tensorrt/TensorRT-8.6.1.6/lib/libnvonnxparser.so
                           /home/robot/tensorrt/TensorRT-8.6.1.6/lib/libnvparsers.so
                           /home/robot/tensorrt/TensorRT-8.6.1.6/lib/libnvinfer_plugin.so
                           ${OpenCV_LIBS})



# 使用cuda
set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
