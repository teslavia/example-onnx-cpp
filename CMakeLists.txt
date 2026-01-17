cmake_minimum_required(VERSION 3.15)
project(M4ResNet)

set(CMAKE_CXX_STANDARD 17)

# 自动探测 Homebrew 前缀 (适配 M4 Pro /opt/homebrew)
find_program(BREW_PROG brew)
if(BREW_PROG)
    execute_process(COMMAND ${BREW_PROG} --prefix OUTPUT_VARIABLE BREW_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
else()
    set(BREW_PREFIX "/usr/local") # Fallback for Intel Macs (optional)
endif()

message(STATUS "Homebrew prefix detected: ${BREW_PREFIX}")

# 1. 设置 OpenCV
set(OpenCV_DIR "${BREW_PREFIX}/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

# 2. 设置 ONNX Runtime (Homebrew 路径)
include_directories(${BREW_PREFIX}/include/onnxruntime)
link_directories(${BREW_PREFIX}/lib)

add_executable(resnet_infer inference.cpp)

target_link_libraries(resnet_infer
    ${OpenCV_LIBS}
    onnxruntime
)

# 解决 macOS RPATH 问题 (确保运行时能找到 dylib)
set_target_properties(resnet_infer PROPERTIES
    BUILD_RPATH "${BREW_PREFIX}/lib"
    INSTALL_RPATH "${BREW_PREFIX}/lib"
)