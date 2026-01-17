这是一个关于 **“使用 ONNX Runtime (C++) 部署 ResNet 模型”** 的工程总结文档。

# ResNet ONNX C++ 高性能部署指南

**目标硬件**：Mac (Apple Silicon)
**核心技术栈**：PyTorch (导出), ONNX Runtime (推理), OpenCV (图像处理), CMake (构建)
**适用场景**：安防监控、工业质检、边缘计算设备上的图像分类/特征提取

------

## 1. 环境准备 (Environment)

利用 macOS 的 **Homebrew** 进行依赖管理，确保所有库均原生支持 ARM64 架构，避免 Rosetta 转译带来的性能损耗。

**终端执行命令：**

```
# 1. 安装基础构建工具和依赖
brew update
brew install cmake opencv onnxruntime

# 2. 验证安装 (可选)
# 确保输出包含 arm64 架构信息
file $(brew --prefix onnxruntime)/lib/libonnxruntime.dylib
```

------

## 2. 核心流程 (Workflow)

### 步骤一：模型导出 (Python)

使用 PyTorch 将训练好的 ResNet-18 导出为标准 ONNX 格式。关键点在于**固化 Batch Norm** 和设置**动态 Batch Size**。

- **关键代码配置**：

  ```
  torch.onnx.export(
      model, 
      dummy_input, 
      "resnet18.onnx",
      input_names=["input_image"], 
      output_names=["class_logits"],
      opset_version=13,
      dynamic_axes={"input_image": {0: "batch_size"}} # 允许变长 Batch
  )
  ```

### 步骤二：C++ 推理实现 (Optimization for M4)

针对多核强大的 CPU 性能和统一内存架构，代码做了以下特定优化：

1. **指针访问 (Pointer Access)**：放弃 OpenCV 的 .at() 慢速访问，使用原始指针遍历像素，大幅提升 HWC -> CHW 的重排速度，命中 CPU 缓存。
2. **零拷贝张量构建 (Zero Copy)**：直接使用 std::vector 的内存地址创建 Ort::Value，避免数据传入推理引擎时的二次拷贝。
3. **多线程设置**：session_options.SetIntraOpNumThreads(4)，利用多核。

- **高性能预处理代码片段**：

  ```
  // 预先获取指针，利用 M4 宽指令集优势顺序写入
  float* ptr_r = input_tensor_values.data();
  float* ptr_g = input_tensor_values.data() + pixels_count;
  float* ptr_b = input_tensor_values.data() + 2 * pixels_count;
  
  for (int h = 0; h < input_h; ++h) {
      const uchar* row_ptr = resized_img.ptr<uchar>(h); // 获取行指针
      for (int w = 0; w < input_w; ++w) {
          // 归一化并拆分通道 (BGR -> RGB planar)
          *ptr_r++ = (row_ptr[0] / 255.0f - mean[0]) / std[0];
          *ptr_g++ = (row_ptr[1] / 255.0f - mean[1]) / std[1];
          *ptr_b++ = (row_ptr[2] / 255.0f - mean[2]) / std[2];
          row_ptr += 3;
      }
  }
  ```

### 步骤三：构建系统 (CMake)

针对 macOS Homebrew 路径 (/opt/homebrew) 和 RPATH 机制编写 CMakeLists.txt，确保编译后的可执行文件能自动找到动态库。

- **CMake 关键配置**：

  ```
  # 自动探测 Homebrew 路径
  find_program(BREW_PROG brew)
  execute_process(COMMAND ${BREW_PROG} --prefix OUTPUT_VARIABLE BREW_PREFIX ...)
  
  # 链接库
  include_directories(${BREW_PREFIX}/include/onnxruntime)
  link_directories(${BREW_PREFIX}/lib)
  
  # 修复 macOS 运行时找不到 dylib 的问题
  set_target_properties(resnet_infer PROPERTIES INSTALL_RPATH "${BREW_PREFIX}/lib")
  ```

------

## 3. 项目结构与运行

建议的项目文件布局：

```bash
Project/
├── CMakeLists.txt      # 构建脚本
├── main.cpp            # C++ 推理主程序
├── export_resnet.py    # Python 导出脚本
├── resnet18.onnx       # (生成) 模型文件
├── dog.jpg             # 测试图片
└── build/              # 编译目录

Project/
├── CMakeLists.txt			# 构建脚本
├── img									# 图片目录
│   └── dog.jpg
├── inference.cpp				# C++ 推理主程序
├── model								# 模型目录
│   ├── out							# 产物
│   │   └── resnet18.onnx
│   └── resnet18.py			# onnx导出脚本
└── readme.md
```

**编译命令**：

```bash
mkdir build && cd build
cmake ..
make
```

**运行命令**：

```bash
./resnet_infer
```

------

## 4. 性能预期 (Performance Note)

在 **Mac** 上运行此方案：

- **预处理耗时**：极低（得益于指针优化）。
- **推理耗时**：ResNet-18 单帧推理通常在 **2ms - 5ms** 之间（CPU 模式）。
- **优势**：相比于配置复杂的 CoreML 或 Metal 后端，ONNX Runtime CPU 版本在 M4 上提供了极佳的**开发效率与运行速度的平衡**，且代码具有跨平台移植性（可直接部署到 Linux/Windows Server）。

## 5. 常见问题排查 (FAQ)

1. **报错 dyld: Library not loaded**：
   - 原因：系统找不到动态库。
   - 解决：确保 CMake 中设置了 INSTALL_RPATH，或者手动设置环境变量 export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH。
2. **推理结果不正确 (概率极低)**：
   - 检查：确保 C++ 中的均值/方差 (mean, std) 与 Python 训练/导出时完全一致。
   - 检查：OpenCV 默认读取为 **BGR**，需转换为 RGB。
3. **想进一步加速**：
   - 虽然多核很快，但如果需要批处理大流量，可以尝试编译 **ONNX Runtime with CoreML EP** (Execution Provider)，利用 Neural Engine (NPU)。但在 ResNet-18 这种小模型上，CPU 纯算往往延迟更低。