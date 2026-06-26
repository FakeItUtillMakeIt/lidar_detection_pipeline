cmake_minimum_required(VERSION 3.18)

# CUDA 编译器和架构必须在 project() 之前设置
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-13.1/bin/nvcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++-9")
set(CMAKE_CUDA_ARCHITECTURES "86")

project(lidar_detection_pipeline
    VERSION 1.0.0
    DESCRIPTION "Modular LiDAR detection pipeline"
    LANGUAGES CXX CUDA
)

# ------------------------------------------------------------------------------
# 1. 选项配置
# ------------------------------------------------------------------------------
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release MinSizeRel RelWithDebInfo.")
option(WITH_ROS2 "Build with ROS2 support" OFF)
option(WITH_OPENCV "Build with OpenCV BEV visualization" ON)

# ------------------------------------------------------------------------------
# 2. 全局编译设置
# ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# 导出 compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CUDA 编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler --expt-relaxed-constexpr")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -Wno-deprecated-declarations")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -Xcompiler -O3")

include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(nlohmann_json)

# ------------------------------------------------------------------------------
# 3. 依赖管理
# ------------------------------------------------------------------------------
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    /usr/local/cuda-13.1/include
    $ENV{TensorRT_Inc}
)

link_directories(
    /usr/local/cuda-13.1/lib64
    $ENV{TensorRT_Lib}
)

# ------------------------------------------------------------------------------
# 4. 核心库构建
# ------------------------------------------------------------------------------
add_library(lidar_core SHARED
    src/core/node.cpp
    src/core/pipeline_manager.cpp
)

set_target_properties(lidar_core PROPERTIES
    LINKER_LANGUAGE CXX
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)

target_include_directories(lidar_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(lidar_core
    PUBLIC
        nlohmann_json::nlohmann_json
)

# ------------------------------------------------------------------------------
# 5. 节点库构建
# ------------------------------------------------------------------------------

# 源节点
set(SOURCE_NODE_SOURCES
    src/nodes/source/reader.cpp
    src/nodes/source/bin_reader.cpp
    src/nodes/source/bin_source_node.cpp
)

# 推理节点 (包含 CUDA 文件)
set(INFER_NODE_SOURCES
    src/nodes/infer/engine.cpp
    src/nodes/infer/detector.cpp
    src/nodes/infer/pointpillar-scatter.cpp
    src/nodes/infer/common/tensorrt.cpp
    src/nodes/infer/lidar-backbone.cu
    src/nodes/infer/lidar-postprocess.cu
    src/nodes/infer/lidar-voxelization.cu
    src/nodes/infer/pillarscatter-kernel.cu
    src/nodes/infer/detection_infer_node.cpp
)

# 输出节点
set(OUTPUT_NODE_SOURCES
    src/nodes/output/output.cpp
    src/nodes/output/file_writer.cpp
    src/nodes/output/file_output_node.cpp
)

# 注册
set(REGISTRY_SOURCES
    src/nodes/registry/node_registry.cpp
)

add_library(lidar_nodes SHARED
    ${SOURCE_NODE_SOURCES}
    ${INFER_NODE_SOURCES}
    ${OUTPUT_NODE_SOURCES}
    ${REGISTRY_SOURCES}
)

target_include_directories(lidar_nodes
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/nodes/registry
        ${CMAKE_CURRENT_SOURCE_DIR}/src/nodes/source
        ${CMAKE_CURRENT_SOURCE_DIR}/src/nodes/infer
        ${CMAKE_CURRENT_SOURCE_DIR}/src/nodes/infer/common
        ${CMAKE_CURRENT_SOURCE_DIR}/src/nodes/output
)

target_link_libraries(lidar_nodes
    PUBLIC
        lidar_core
    PRIVATE
        libnvinfer.so
        libnvinfer_plugin.so
)

# ------------------------------------------------------------------------------
# 6. 主程序构建
# ------------------------------------------------------------------------------
add_executable(lidar_app src/main.cpp)

target_link_libraries(lidar_app
    PRIVATE
        lidar_core
        lidar_nodes
        libnvinfer.so
        libnvinfer_plugin.so
)

set_target_properties(lidar_app PROPERTIES
    LINKER_LANGUAGE CXX
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)

# ------------------------------------------------------------------------------
# 7. 输出构建摘要
# ------------------------------------------------------------------------------
message(STATUS "")
message(STATUS "=== lidar_detection_pipeline Configuration Summary ===")
message(STATUS "  Build type:           ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard:         ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA Compiler:        ${CMAKE_CUDA_COMPILER}")
message(STATUS "  CUDA Architectures:   ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  ROS2 support:         ${WITH_ROS2}")
message(STATUS "  OpenCV support:       ${WITH_OPENCV}")
message(STATUS "  Install prefix:       ${CMAKE_INSTALL_PREFIX}")
message(STATUS "=====================================================")
