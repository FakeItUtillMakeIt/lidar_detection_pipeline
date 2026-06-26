// include/lidar_core/nodes/i_attribute_node.h
#pragma once

#include "lidar_core/core/node.h"

namespace lidar_core {
namespace nodes {

// 属性计算节点接口
// 计算跟踪目标的运动属性（速度、航向角等）
class IAttributeNode : public core::Node {
public:
    using core::Node::Node;

    // 设置时间间隔(秒)，用于速度计算
    virtual void setTimeInterval(float dt) = 0;

    // 获取时间间隔
    virtual float getTimeInterval() const = 0;
};

} // namespace nodes
} // namespace lidar_core
