cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_COMPILER "/usr/bin/g++-9")
project(lidar_detection_pipeline)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

# CUDA arch
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    -gencode arch=compute_86,code=compute_86
    -allow-unsupported-compiler
    --expt-relaxed-constexpr
)
set(CUDA_NVCC_FLAGS_RELEASE "-Werror=all-warnings -Xcompiler -O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -Wno-deprecated-declarations")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    $ENV{CUDA_Inc}
    $ENV{TensorRT_Inc}
)

link_directories(
    $ENV{CUDA_Lib}
    $ENV{TensorRT_Lib}
)

# ============================================================
# 1. liblidar_core.so — Detection core (Voxelization + Backbone + PostProcess)
# ============================================================
file(GLOB CORE_CUDA_FILES
    src/core/*.cu
    src/core/detector.cpp
    src/core/pointpillar-scatter.cpp
    src/common/tensorrt.cpp
)

cuda_add_library(lidar_core SHARED ${CORE_CUDA_FILES})
target_link_libraries(lidar_core libnvinfer.so libnvinfer_plugin.so)

# ============================================================
# 2. libpipeline.so — Pipeline layer (Reader + Engine + Output)
# ============================================================
set(PIPELINE_SOURCES
    src/pipeline/reader.cpp
    src/pipeline/bin_reader.cpp
    src/pipeline/velodyne_reader.cpp
    src/pipeline/engine.cpp
    src/pipeline/output.cpp
    src/pipeline/file_writer.cpp
    src/pipeline/callback_output.cpp
    src/pipeline/bev_visualizer.cpp
)

option(WITH_ROS2 "Build with ROS2 support" OFF)
if(WITH_ROS2)
    find_package(rclcpp REQUIRED)
    find_package(visualization_msgs REQUIRED)
    list(APPEND PIPELINE_SOURCES src/pipeline/ros2_output.cpp)
endif()

option(WITH_OPENCV "Build with OpenCV BEV visualization" ON)
if(WITH_OPENCV)
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

add_library(pipeline SHARED ${PIPELINE_SOURCES})
target_link_libraries(pipeline lidar_core)
if(WITH_OPENCV AND OpenCV_FOUND)
    target_link_libraries(pipeline ${OpenCV_LIBS})
    target_compile_definitions(pipeline PRIVATE -DWITH_OPENCV)
endif()
if(WITH_ROS2)
    target_link_libraries(pipeline ${rclcpp_TARGETS} visualization_msgs::visualization_msgs)
    target_compile_definitions(pipeline PRIVATE -DWITH_ROS2)
endif()

# ============================================================
# 3. lidar_app — Main entry
# ============================================================
add_executable(lidar_app src/app.cpp)
target_link_libraries(lidar_app pipeline lidar_core)
if(WITH_OPENCV AND OpenCV_FOUND)
    target_link_libraries(lidar_app ${OpenCV_LIBS})
endif()
