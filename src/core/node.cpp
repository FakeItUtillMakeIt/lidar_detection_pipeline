// src/core/node.cpp
#include "lidar_core/core/node.h"
#include "lidar_core/core/pipeline.h"
#include <iostream>

namespace lidar_core {
namespace core {

void Node::broadcast(std::shared_ptr<BasePacket> packet) {
    for (size_t i = 0; i < downstreams_.size(); ++i) {
        if (auto downstream = downstreams_[i].lock()) {
            if (i + 1 < downstreams_.size()) {
                downstream->pushData(packet->clone());
            } else {
                downstream->pushData(packet);
            }
        }
    }
}

} // namespace core
} // namespace lidar_core
