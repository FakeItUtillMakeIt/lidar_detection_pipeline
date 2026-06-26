// src/nodes/track/ocsort_adapter.h
#pragma once

#include "lidar_core/core/packet.h"
#include <vector>
#include <memory>

namespace lidar_core {
namespace nodes {

class OCSortAdapter {
public:
    OCSortAdapter();
    ~OCSortAdapter();

    bool init(float det_thresh = 0.3f,
              int max_age = 30,
              int min_hits = 3,
              float iou_threshold = 0.3f,
              int delta_t = 3,
              float inertia = 0.2f,
              bool use_byte = false);

    // 在原始检测上标注 track_id，不修改其他字段
    void update(std::vector<core::Detection>& detections);

    int getActiveTrackCount() const;
    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace nodes
} // namespace lidar_core
