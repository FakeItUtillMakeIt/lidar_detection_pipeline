// src/nodes/track/ocsort_adapter.h
#pragma once

#include "lidar_core/core/packet.h"
#include <vector>
#include <memory>

namespace lidar_core {
namespace nodes {

// OCSort适配器，用于3D LiDAR BEV跟踪
class OCSortAdapter {
public:
    OCSortAdapter();
    ~OCSortAdapter();

    // 初始化跟踪器
    bool init(float det_thresh = 0.3f,
              int max_age = 30,
              int min_hits = 3,
              float iou_threshold = 0.3f,
              int delta_t = 3,
              float inertia = 0.2f,
              bool use_byte = false);

    // 更新跟踪器，返回带track_id的检测结果
    std::vector<core::Detection> update(const std::vector<core::Detection>& detections);

    // 获取活跃跟踪数量
    int getActiveTrackCount() const;

    // 重置跟踪器
    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace nodes
} // namespace lidar_core
