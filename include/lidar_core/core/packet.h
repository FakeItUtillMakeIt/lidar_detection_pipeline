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
};

} // namespace core
} // namespace lidar_core
