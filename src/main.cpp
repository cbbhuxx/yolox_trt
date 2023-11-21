
// tensorRT include
// 编译用的头文件
#include <NvInfer.h>

// 推理用的运行时头文件
#include <NvInferRuntime.h>

// cuda include
#include <cuda_runtime.h>

// system include
#include <stdio.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <functional>
#include <unistd.h>
#include <opencv2/opencv.hpp>

#include <common/ilogger.hpp>
#include <builder/trt_builder.hpp>
#include <app_yolo/yolo.hpp>

using namespace std;

static const char* cocolabels[] = {
//        "person", "bicycle", "car", "motorcycle", "airplane",
//        "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
//        "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
//        "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
//        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
//        "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
//        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
//        "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
//        "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
//        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
//        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
//        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
//        "scissors", "teddy bear", "hair drier", "toothbrush"

    "red circle", "yellow circle"
};

static bool exists(const string& path){

#ifdef _WIN32
    return ::PathFileExistsA(path.c_str());
#else
    return access(path.c_str(), R_OK) == 0;
#endif
}

// 上一节的代码
static bool build_model(){

    if(exists("../workspace/yolox_s_ring.trtmodel")){
        printf("../workspace/yolox_s_ring.trtmodel has exists.\n");
        return true;
    }

    //SimpleLogger::set_log_level(SimpleLogger::LogLevel::Verbose);
    TRT::compile(
        TRT::Mode::FP32,
        1,
        "../workspace/yolox_s_ring.onnx",
        "../workspace/yolox_s_ring.trtmodel"
    );
    INFO("Done.");
    return true;
}

static void inference(){

//    auto image = cv::imread("../workspace/rq.jpg");
    auto yolov5 = Yolo::create_infer("../workspace/yolox_s_ring.trtmodel", Yolo::Type::V5, 0, 0.25, 0.45);
    cv::VideoCapture cap(0);
    while (true)
    {
        // 读取摄像头数据
        cv::Mat image;
        cap >> image;
        // 获取起始时间点
        auto start = std::chrono::high_resolution_clock::now();
        auto boxes = yolov5->commit(image).get();
        // 获取结束时间点
        auto end = std::chrono::high_resolution_clock::now();
        // 计算时间差，单位为微秒
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        // 计算帧率，单位为帧/秒
        double fps = 1000000.0 / duration.count();
        std::cout << fps << std::endl;

        // 在图像上绘制帧率
        cv::putText(image, "FPS: " + std::to_string(fps), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

        for(auto& box : boxes){
            cv::Scalar color(0, 255, 0);
            cv::rectangle(image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), color, 3);

            auto name      = cocolabels[box.class_label];
            auto caption   = cv::format("%s %.2f", name, box.confidence);
            int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
            cv::rectangle(image, cv::Point(box.left-3, box.top-33), cv::Point(box.left + text_width, box.top), color, -1);
            cv::putText(image, caption, cv::Point(box.left, box.top-5), 0, 1, cv::Scalar::all(0), 2, 16);
        }
//        cv::imwrite("../workspace/image-draw.jpg", image);
        cv::imshow("frame", image);

        // 等待按键，退出循环
        if (cv::waitKey(1) >= 0)
            break;
    }
}

int main(){

    // 新的实现
    if(!build_model()){
        return -1;
    }
    inference();
    return 0;
}