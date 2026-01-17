#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

// 辅助：计算 Softmax
void softmax(std::vector<float>& input) {
    float max_val = *std::max_element(input.begin(), input.end());
    float sum = 0.0f;
    for (auto& x : input) x = std::exp(x - max_val);
    for (const auto& x : input) sum += x;
    for (auto& x : input) x /= sum;
}

int main() {
    // --------------------------------------------------------
    // 1. 初始化 ONNX Runtime (针对多核 CPU 优化配置)
    // --------------------------------------------------------
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "M4_ResNet");
    Ort::SessionOptions session_options;
    
    // 开启所有图优化
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // 多核 CPU，适当增加线程数（通常设为 1 或 4 效果较好，避免过度上下文切换）
    session_options.SetIntraOpNumThreads(4);

    const char* model_path = "../model/out/resnet18.onnx";
    Ort::Session session(env, model_path, session_options);

    // --------------------------------------------------------
    // 2. 图像读取与快速预处理 (Pointer access)
    // --------------------------------------------------------
    cv::Mat img = cv::imread("../img/dog.jpg");
    if (img.empty()) { std::cerr << "Error: Image not found!" << std::endl; return -1; }

    // ResNet 标准输入尺寸
    const int input_h = 224;
    const int input_w = 224;
    
    // A. Resize
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(input_w, input_h));
    
    // B. Convert BGR -> RGB
    cv::cvtColor(resized_img, resized_img, cv::COLOR_BGR2RGB);

    // 准备数据容器 (NCHW)
    std::vector<float> input_tensor_values(1 * 3 * input_h * input_w);
    
    // 归一化参数 (ImageNet Standard)
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std[3]  = {0.229f, 0.224f, 0.225f};

    // C. 优化的 HWC -> CHW 转换 (使用指针遍历，而非 .at())
    // M4 Pro 的 CPU 缓存很大，这种连续内存写入非常快
    float* ptr_r = input_tensor_values.data();
    float* ptr_g = input_tensor_values.data() + input_h * input_w;
    float* ptr_b = input_tensor_values.data() + 2 * input_h * input_w;

    for (int h = 0; h < input_h; ++h) {
        // 获取 OpenCV 图像当前行的指针 (uint8)
        const uchar* row_ptr = resized_img.ptr<uchar>(h);
        for (int w = 0; w < input_w; ++w) {
            // 指针偏移: RGBRGB...
            // row_ptr[0] is R, row_ptr[1] is G, row_ptr[2] is B
            *ptr_r++ = (row_ptr[0] / 255.0f - mean[0]) / std[0];
            *ptr_g++ = (row_ptr[1] / 255.0f - mean[1]) / std[1];
            *ptr_b++ = (row_ptr[2] / 255.0f - mean[2]) / std[2];
            
            row_ptr += 3; // 移动到下一个像素
        }
    }

    // --------------------------------------------------------
    // 3. 推理
    // --------------------------------------------------------
    std::vector<int64_t> input_shape = {1, 3, input_h, input_w};
    
    // 创建 MemoryInfo
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    // 创建 Tensor (Zero Copy)
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(),
        input_shape.data(), input_shape.size()
    );

    const char* input_names[] = {"input_image"};
    const char* output_names[] = {"class_logits"};

    // 计时开始 (可选)
    // auto start = std::chrono::high_resolution_clock::now();

    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1
    );

    // --------------------------------------------------------
    // 4. 解析结果
    // --------------------------------------------------------
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    std::vector<float> result(output_data, output_data + 1000); // 1000 classes
    softmax(result);

    auto max_it = std::max_element(result.begin(), result.end());
    int pred_id = std::distance(result.begin(), max_it);

    std::cout << "[M4 Pro Inference]" << std::endl;
    std::cout << "Class ID: " << pred_id << std::endl;
    std::cout << "Prob    : " << *max_it << std::endl;

    return 0;
}