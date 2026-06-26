// src/nodes/infer/detection_infer_node.cpp
#include "detection_infer_node.h"
#include "node_factory.h"
#include <iostream>

namespace lidar_core {
namespace nodes {

DetectionInferNode::DetectionInferNode() 
    : IInferNode("DetectionInferNode")
    , score_thresh_(0.1f)
    , nms_thresh_(0.01f) {
}

DetectionInferNode::~DetectionInferNode() {
    stop();
}

bool DetectionInferNode::loadModel(const std::string& model_path) {
    model_path_ = model_path;
    
    // 创建引擎配置
    pipeline::EngineConfig config;
    config.model_path = model_path;
    config.score_thresh = score_thresh_;
    config.nms_thresh = nms_thresh_;
    config.async_mode = false;

    // 创建引擎
    engine_ = std::make_shared<pipeline::PointPillarsEngine>(config);
    if (!engine_->init()) {
        std::cerr << "[DetectionInferNode] Failed to init engine" << std::endl;
        return false;
    }

    std::cout << "[DetectionInferNode] Model loaded: " << model_path << std::endl;
    return true;
}

void DetectionInferNode::setModelParams(const std::string& model_path, 
                                        float score_thresh,
                                        float nms_thresh) {
    model_path_ = model_path;
    score_thresh_ = score_thresh;
    nms_thresh_ = nms_thresh;
}

std::string DetectionInferNode::getModelPath() const {
    return model_path_;
}

bool DetectionInferNode::start() {
    if (running_)
        return true;

    if (!engine_) {
        std::cerr << "[DetectionInferNode] Engine not initialized" << std::endl;
        return false;
    }

    running_ = true;
    std::cout << "[DetectionInferNode] Started" << std::endl;
    return true;
}

void DetectionInferNode::stop() {
    if (!running_)
        return;
    running_ = false;
    
    if (engine_) {
        engine_->shutdown();
    }
    std::cout << "[DetectionInferNode] Stopped" << std::endl;
}

void DetectionInferNode::pushData(std::shared_ptr<core::BasePacket> packet) {
    auto cloud_packet = std::dynamic_pointer_cast<core::PointCloudPacket>(packet);
    if (!cloud_packet) {
        std::cerr << "[DetectionInferNode] Invalid packet type" << std::endl;
        return;
    }

    // 转换点云数据
    pipeline::PointCloud cloud;
    cloud.frame_id = cloud_packet->frame_id;
    cloud.timestamp_ns = cloud_packet->timestamp_ns;
    cloud.points.reserve(cloud_packet->points.size());
    
    for (const auto& pt : cloud_packet->points) {
        pipeline::PointXYZI point;
        point.x = pt.x;
        point.y = pt.y;
        point.z = pt.z;
        point.intensity = pt.intensity;
        cloud.points.push_back(point);
    }

    // 执行推理
    auto detections = engine_->detect(cloud);

    // 创建检测结果包
    auto det_packet = std::make_shared<core::DetectionPacket>();
    det_packet->frame_id = cloud_packet->frame_id;
    det_packet->timestamp_ns = cloud_packet->timestamp_ns;
    det_packet->detections.reserve(detections.size());
    
    for (const auto& det : detections) {
        core::Detection detection;
        detection.x = det.x;
        detection.y = det.y;
        detection.z = det.z;
        detection.w = det.w;
        detection.l = det.l;
        detection.h = det.h;
        detection.rt = det.rt;
        detection.class_id = det.class_id;
        detection.score = det.score;
        detection.frame_id = det.frame_id;
        det_packet->detections.push_back(detection);
    }

    // 广播给下游节点
    broadcast(det_packet);
}

// 注册节点
REGISTER_NODE("detection_infer", DetectionInferNode)

} // namespace nodes
} // namespace lidar_core
