// src/nodes/registry/node_factory.cpp
#include "node_factory.h"
#include <iostream>
#include <unordered_map>

namespace lidar_core {
namespace core {

struct NodeFactory::Impl {
    std::unordered_map<std::string, NodeCreator> creators;
};

NodeFactory& NodeFactory::instance() {
    static NodeFactory factory;
    return factory;
}

NodeFactory::NodeFactory() : impl_(std::make_unique<Impl>()) {}

void NodeFactory::registerCreator(const std::string& type, NodeCreator creator) {
    impl_->creators[type] = std::move(creator);
}

std::shared_ptr<Node> NodeFactory::create(const std::string& type, const nlohmann::json& params) {
    auto it = impl_->creators.find(type);
    if (it != impl_->creators.end()) {
        return it->second(params);
    }
    std::cerr << "[NodeFactory] Node type not found: " << type << std::endl;
    return nullptr;
}

} // namespace core
} // namespace lidar_core
