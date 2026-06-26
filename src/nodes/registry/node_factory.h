// src/nodes/registry/node_factory.h
#pragma once

#include "lidar_core/core/node.h"
#include <functional>
#include <memory>
#include <string>
#include <nlohmann/json.hpp>

namespace lidar_core {
namespace core {

using NodeCreator = std::function<std::shared_ptr<Node>(const nlohmann::json&)>;

class NodeFactory {
public:
    static NodeFactory& instance();

    void registerCreator(const std::string& type, NodeCreator creator);
    std::shared_ptr<Node> create(const std::string& type, const nlohmann::json& params);

private:
    NodeFactory();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

template<typename T>
class NodeRegistrar {
public:
    NodeRegistrar(const std::string& type) {
        NodeFactory::instance().registerCreator(type, [](const nlohmann::json&) -> std::shared_ptr<Node> {
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
