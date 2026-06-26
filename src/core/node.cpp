// src/core/node.cpp
#include "lidar_core/core/node.h"
#include "lidar_core/core/pipeline.h"
#include <iostream>

namespace lidar_core {
namespace core {

void Node::broadcast(std::shared_ptr<BasePacket> packet) {
    for (auto& weak_downstream : downstreams_) {
        if (auto downstream = weak_downstream.lock()) {
            downstream->pushData(packet);
        }
    }
}

} // namespace core
} // namespace lidar_core
