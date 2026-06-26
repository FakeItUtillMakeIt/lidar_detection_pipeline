// include/lidar_core/nodes/i_tracker_node.h
#pragma once

#include "lidar_core/core/node.h"
#include <string>

namespace lidar_core {
namespace nodes {

// 跟踪器类型
enum class TrackerType {
    OCSORT = 0,
    SORT = 1,
};

// OCSort配置
struct OCSortConfig {
    float det_thresh = 0.3f;
    int max_age = 30;
    int min_hits = 3;
    float iou_threshold = 0.3f;
    int delta_t = 3;
    std::string asso_func = "iou";
    float inertia = 0.2f;
    bool use_byte = false;
};

// 跟踪节点接口
class ITrackerNode : public core::Node {
public:
    using core::Node::Node;

    // 设置跟踪器类型
    virtual void setTrackerType(TrackerType type) = 0;
    virtual TrackerType getTrackerType() const = 0;

    // 设置OCSort配置
    virtual void setOCSortConfig(const OCSortConfig& config) = 0;

    // 获取活跃跟踪数量
    virtual int getActiveTrackCount() const = 0;
};

} // namespace nodes
} // namespace lidar_core
