// src/nodes/infer/detection_infer_node.h
#pragma once

#include "lidar_core/nodes/i_infer_node.h"
#include "engine.hpp"
#include <memory>

namespace lidar_core {
namespace nodes {

class DetectionInferNode : public IInferNode {
public:
    DetectionInferNode();
    ~DetectionInferNode() override;

    // IInferNode 接口
    bool loadModel(const std::string& model_path) override;
    void setModelParams(const std::string& model_path, 
                        float score_thresh = 0.1f,
                        float nms_thresh = 0.01f) override;
    std::string getModelPath() const override;

    // Node 接口
    bool start() override;
    void stop() override;
    void pushData(std::shared_ptr<core::BasePacket> packet) override;

private:
    std::string model_path_;
    float score_thresh_;
    float nms_thresh_;
    std::shared_ptr<pipeline::PointPillarsEngine> engine_;
};

} // namespace nodes
} // namespace lidar_core
