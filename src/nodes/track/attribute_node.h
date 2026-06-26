// src/nodes/track/attribute_node.h
#pragma once

#include "lidar_core/nodes/i_attribute_node.h"
#include <unordered_map>

namespace lidar_core {
namespace nodes {

class AttributeNode : public IAttributeNode {
public:
    AttributeNode();
    ~AttributeNode() override;

    // IAttributeNode 接口
    void setTimeInterval(float dt) override;
    float getTimeInterval() const override;

    // Node 接口
    bool start() override;
    void stop() override;
    void pushData(std::shared_ptr<core::BasePacket> packet) override;

private:
    // 计算速度和航向角
    void computeAttributes(core::Detection& det);

    float dt_ = 0.1f;  // 默认100ms
    
    // 历史位置记录 (track_id -> last position)
    struct TrackHistory {
        float x, y;
        uint64_t timestamp_ns;
        bool valid = false;
    };
    std::unordered_map<int, TrackHistory> history_;
};

} // namespace nodes
} // namespace lidar_core
