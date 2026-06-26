#ifndef __PIPELINE_TYPES_HPP__
#define __PIPELINE_TYPES_HPP__

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace pipeline {

// ============================================================
// Point Cloud
// ============================================================
struct PointXYZI {
    float x, y, z, intensity;
};

struct PointCloud {
    std::vector<PointXYZI> points;
    uint64_t timestamp_ns = 0;   // nanoseconds
    uint64_t frame_id = 0;
};

// ============================================================
// Detection Result
// ============================================================
struct Detection {
    float x, y, z;     // center
    float w, l, h;      // width, length, height
    float rt;            // rotation angle (yaw)
    int class_id;        // 0=Car, 1=Pedestrian, 2=Cyclist
    float score;
    uint64_t frame_id = 0;

    // 跟踪字段
    int track_id = -1;       // 跟踪ID (-1=未跟踪)
    int track_age = 0;       // 跟踪生命周期(帧数)

    // 运动属性
    float vx = 0;            // x方向速度 (m/s)
    float vy = 0;            // y方向速度 (m/s)
    float speed = 0;         // 速度大小 (m/s)
    float heading = 0;       // 航向角 (rad)
};

// ============================================================
// Enums
// ============================================================
enum class ReaderType {
    BIN_FILE,
    VELODYNE_UDP,
};

enum class OutputType {
    FILE,
    CALLBACK,
    ROS2,
    BEV_VISUALIZER,
};

// ============================================================
// Config Structures
// ============================================================
struct ReaderConfig {
    ReaderType type = ReaderType::BIN_FILE;
    std::string input_path;       // file or directory path
    int velodyne_port = 2368;
};

struct EngineConfig {
    std::string model_path = "../model/pointpillar.plan";
    float min_range[3] = {0.0f, -39.68f, -3.0f};
    float max_range[3] = {69.12f, 39.68f, 1.0f};
    float voxel_size[3] = {0.16f, 0.16f, 4.0f};
    int max_voxels = 40000;
    int max_points_per_voxel = 32;
    int max_points = 300000;
    float score_thresh = 0.1f;
    float nms_thresh = 0.01f;
    bool async_mode = false;
    int num_workers = 2;
};

struct OutputConfig {
    OutputType type = OutputType::FILE;
    std::string output_dir = "../out";
};

// ============================================================
// Callback type
// ============================================================
using DetectionCallback = std::function<void(const std::vector<Detection>&, uint64_t frame_id)>;

}  // namespace pipeline

#endif  // __PIPELINE_TYPES_HPP__
