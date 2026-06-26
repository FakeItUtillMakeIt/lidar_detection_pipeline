// src/nodes/registry/node_factory.h
#pragma once

#include "lidar_core/core/node.h"
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <nlohmann/json.hpp>

namespace lidar_core {
namespace core {

using NodeCreator = std::function<std::shared_ptr<Node>(const nlohmann::json&)>;

class NodeFactory {
public:
    static NodeFactory& instance() {
        static NodeFactory factory;
        return factory;
    }

    void registerCreator(const std::string& type, NodeCreator creator) {
        creators_[type] = std::move(creator);
        std::cout << "[NodeFactory] Registered node type: " << type << std::endl;
    }

    std::shared_ptr<Node> create(const std::string& type, const nlohmann::json& params) {
        std::cout << "[NodeFactory] Creating node type: " << type << std::endl;
        std::cout << "[NodeFactory] Registered node types count: " << creators_.size() << std::endl;
        
        for (const auto& [key, value] : creators_) {
            std::cout << "[NodeFactory]   Registered: " << key << std::endl;
        }

        auto it = creators_.find(type);
        if (it != creators_.end()) {
            std::cout << "[NodeFactory] Found creator for type: " << type << std::endl;
            return it->second(params);
        }

        std::cerr << "[NodeFactory] Node type not found: " << type << std::endl;
        return nullptr;
    }

private:
    NodeFactory() = default;
    std::unordered_map<std::string, NodeCreator> creators_;
};

template<typename T>
class NodeRegistrar {
public:
    NodeRegistrar(const std::string& type) {
        NodeFactory::instance().registerCreator(type, [](const nlohmann::json& params) -> std::shared_ptr<Node> {
            return std::make_shared<T>();
        });
    }
};

} // namespace core
} // namespace lidar_core

// 注册宏
#define REGISTER_NODE(type, full_class_name) \
    namespace { \
        static lidar_core::core::NodeRegistrar<full_class_name> _node_registrar(type); \
    }
