// src/nodes/track/attribute_node.cpp
#include "attribute_node.h"
#include "src/nodes/registry/node_factory.h"
#include <cmath>
#include <iostream>

namespace lidar_core {
namespace nodes {

AttributeNode::AttributeNode() : IAttributeNode("AttributeNode") {
}

AttributeNode::~AttributeNode() {
    stop();
}

void AttributeNode::setTimeInterval(float dt) {
    dt_ = dt;
}

float AttributeNode::getTimeInterval() const {
    return dt_;
}

bool AttributeNode::start() {
    if (running_)
        return true;

    running_ = true;
    std::cout << "[AttributeNode] Started, dt=" << dt_ << "s" << std::endl;
    return true;
}

void AttributeNode::stop() {
    if (!running_)
        return;
    running_ = false;
    history_.clear();
    std::cout << "[AttributeNode] Stopped" << std::endl;
}

void AttributeNode::pushData(std::shared_ptr<core::BasePacket> packet) {
    auto det_packet = std::dynamic_pointer_cast<core::DetectionPacket>(packet);
    if (!det_packet) {
        std::cerr << "[AttributeNode] Invalid packet type" << std::endl;
        return;
    }

    // 计算每个检测目标的属性
    for (auto& det : det_packet->detections) {
        computeAttributes(det);
    }

    // 广播给下游节点
    broadcast(det_packet);
}

void AttributeNode::computeAttributes(core::Detection& det) {
    int track_id = det.track_id;

    if (track_id < 0) {
        // 未跟踪的目标，无法计算速度
        det.vx = 0;
        det.vy = 0;
        det.speed = 0;
        det.heading = det.rt;  // 使用检测框的yaw角
        return;
    }

    // 查找历史记录
    auto it = history_.find(track_id);
    if (it != history_.end() && it->second.valid) {
        // 计算速度 (m/s)
        float dx = det.x - it->second.x;
        float dy = det.y - it->second.y;
        
        // 使用配置的时间间隔
        det.vx = dx / dt_;
        det.vy = dy / dt_;
        
        // 计算速度大小
        det.speed = std::sqrt(det.vx * det.vx + det.vy * det.vy);
        
        // 计算航向角 (从速度方向)
        if (det.speed > 0.1f) {  // 速度大于0.1m/s时使用速度方向
            det.heading = std::atan2(det.vy, det.vx);
        } else {
            det.heading = det.rt;  // 速度太小时使用检测框yaw角
        }
    } else {
        // 首次出现，无法计算速度
        det.vx = 0;
        det.vy = 0;
        det.speed = 0;
        det.heading = det.rt;
    }

    // 更新历史记录
    history_[track_id] = {det.x, det.y, 0, true};
}

// 注册节点
REGISTER_NODE("attribute", AttributeNode)

} // namespace nodes
} // namespace lidar_core
