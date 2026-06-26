// include/lidar_core/nodes/i_output_node.h
#pragma once

#include "lidar_core/core/node.h"
#include <string>

namespace lidar_core {
namespace nodes {

// 输出节点接口
class IOutputNode : public core::Node {
public:
    using core::Node::Node;
    
    // 设置输出目录
    virtual void setOutputDir(const std::string& dir) = 0;
    
    // 设置输出类型
    virtual void setOutputType(const std::string& type) = 0;
    
    // 获取输出目录
    virtual std::string getOutputDir() const = 0;
    
    // 关闭输出
    virtual void close() = 0;
};

} // namespace nodes
} // namespace lidar_core
