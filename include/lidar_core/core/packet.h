// include/lidar_core/core/packet.h
#pragma once

#include <memory>
#include <vector>
#include <cstdint>

namespace lidar_core {
namespace core {

// 点云点结构
struct PointXYZI {
    float x, y, z, intensity;
};

// 检测结果结构
struct Detection {
    float x, y, z;      // 中心点
    float w, l, h;       // 宽度、长度、高度
    float rt;            // 旋转角(yaw)
    int class_id;        // 0=Car, 1=Pedestrian, 2=Cyclist
    float score;
    uint64_t frame_id = 0;

    // 跟踪字段
    int track_id = -1;       // 跟踪ID (-1=未跟踪)
    int track_age = 0;       // 跟踪生命周期(帧数)
    bool track_active = false; // 跟踪是否活跃

    // 运动属性
    float vx = 0;            // x方向速度 (m/s)
    float vy = 0;            // y方向速度 (m/s)
    float speed = 0;         // 速度大小 (m/s)
    float heading = 0;       // 航向角 (rad)
};

// 数据包基类
class BasePacket {
public:
    virtual ~BasePacket() = default;
    uint64_t frame_id = 0;
    uint64_t timestamp_ns = 0;
};

// 点云数据包
class PointCloudPacket : public BasePacket {
public:
    std::vector<PointXYZI> points;
};

// 检测结果数据包
class DetectionPacket : public BasePacket {
public:
    std::vector<Detection> detections;
    std::vector<PointXYZI> cloud_points;  // 携带点云用于BEV可视化
};

// 规划结果数据包
class PlanningPacket : public BasePacket {
public:
    std::vector<Detection> obstacles;              // 规划考虑的障碍物
    std::vector<PointXYZI> cloud_points;           // 携带点云用于可视化
    
    // 规划轨迹
    struct PathPoint {
        float x = 0.0f;           // x位置 (m)
        float y = 0.0f;           // y位置 (m)
        float z = 0.0f;           // z位置 (m)
        float heading = 0.0f;     // 航向角 (rad)
        float curvature = 0.0f;   // 曲率 (1/m)
        float speed = 0.0f;       // 速度 (m/s)
        float acceleration = 0.0f; // 加速度 (m/s²)
        float relative_time = 0.0f; // 相对时间 (s)
    };
    std::vector<PathPoint> trajectory;             // 规划轨迹
    
    // 规划状态
    bool is_feasible = false;                      // 轨迹是否可行
    std::string planning_status;                   // 规划状态信息
    uint64_t planning_timestamp_ns = 0;           // 规划时间戳
};

} // namespace core
} // namespace lidar_core
