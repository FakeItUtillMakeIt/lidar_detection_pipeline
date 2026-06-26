// src/nodes/track/tracker_node.cpp
#include "3rd_party/log_mgr/log_mgr.h"
#include "tracker_node.h"
#include "node_factory.h"
#include <iostream>

namespace lidar_core {
namespace nodes {

TrackerNode::TrackerNode() : ITrackerNode("TrackerNode") {
}

TrackerNode::~TrackerNode() {
    stop();
}

void TrackerNode::setTrackerType(TrackerType type) {
    tracker_type_ = type;
}

TrackerType TrackerNode::getTrackerType() const {
    return tracker_type_;
}

void TrackerNode::setOCSortConfig(const OCSortConfig& config) {
    ocsort_config_ = config;
}

int TrackerNode::getActiveTrackCount() const {
    if (adapter_) {
        return adapter_->getActiveTrackCount();
    }
    return 0;
}

bool TrackerNode::start() {
    if (running_)
        return true;

    // 创建适配器
    if (tracker_type_ == TrackerType::OCSORT) {
        adapter_ = std::make_unique<OCSortAdapter>();
        if (!adapter_->init(
                ocsort_config_.det_thresh,
                ocsort_config_.max_age,
                ocsort_config_.min_hits,
                ocsort_config_.iou_threshold,
                ocsort_config_.delta_t,
                ocsort_config_.inertia,
                ocsort_config_.use_byte)) {
            LOG_ERROR_FMT("[TrackerNode] Failed to init OCSort adapter");
            return false;
        }
    } else {
        LOG_ERROR_FMT("[TrackerNode] Unsupported tracker type");
        return false;
    }

    running_ = true;
    
    LOG_INFO_FMT("[TrackerNode] Started with {}", (tracker_type_ == TrackerType::OCSORT ? "OCSort" : "Unknown"));   
    return true;
}

void TrackerNode::stop() {
    if (!running_)
        return;
    running_ = false;
    
    if (adapter_) {
        adapter_->reset();
    }
    LOG_INFO_FMT("[TrackerNode] Stopped");
}

void TrackerNode::pushData(std::shared_ptr<core::BasePacket> packet) {
    auto det_packet = std::dynamic_pointer_cast<core::DetectionPacket>(packet);
    if (!det_packet) {
        LOG_ERROR_FMT("[TrackerNode] Invalid packet type");
        return;
    }

    if (!adapter_) {
        LOG_ERROR_FMT("[TrackerNode] Adapter not initialized");
        return;
    }

    // 运行跟踪 - 直接在原始检测上标注track_id
    adapter_->update(det_packet->detections);

    // 广播给下游节点
    broadcast(det_packet);
}

// 注册节点
REGISTER_NODE("tracker", TrackerNode)

} // namespace nodes
} // namespace lidar_core
