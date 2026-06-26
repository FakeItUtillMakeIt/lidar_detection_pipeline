// include/lidar_core/nodes/i_infer_node.h
#pragma once

#include "lidar_core/core/node.h"
#include <string>

namespace lidar_core {
namespace nodes {

// 推理节点接口
class IInferNode : public core::Node {
public:
    using core::Node::Node;

    // 加载模型文件
    virtual bool loadModel(const std::string& model_path) = 0;

    // 设置模型参数
    virtual void setModelParams(const std::string& model_path, 
                                float score_thresh = 0.1f,
                                float nms_thresh = 0.01f) = 0;
    
    // 获取模型路径
    virtual std::string getModelPath() const = 0;
};

} // namespace nodes
} // namespace lidar_core
