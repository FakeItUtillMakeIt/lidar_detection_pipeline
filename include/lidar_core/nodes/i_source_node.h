// include/lidar_core/nodes/i_source_node.h
#pragma once

#include "lidar_core/core/node.h"
#include <string>

namespace lidar_core {
namespace nodes {

// 数据源节点接口
class ISourceNode : public core::Node {
public:
    using core::Node::Node;
    
    // 设置输入路径
    virtual void setInputPath(const std::string& path) = 0;
    
    // 设置输入类型
    virtual void setInputType(const std::string& type) = 0;
    
    // 获取输入路径
    virtual std::string getInputPath() const = 0;
};

} // namespace nodes
} // namespace lidar_core
