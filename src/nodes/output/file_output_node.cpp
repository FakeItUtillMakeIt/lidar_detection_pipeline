// src/nodes/output/file_output_node.cpp
#include "file_output_node.h"
#include "node_factory.h"
#include <iostream>

namespace lidar_core {
namespace nodes {

FileOutputNode::FileOutputNode() : IOutputNode("FileOutputNode") {
}

FileOutputNode::~FileOutputNode() {
    close();
}

void FileOutputNode::setOutputDir(const std::string& dir) {
    output_dir_ = dir;
}

void FileOutputNode::setOutputType(const std::string& type) {
    output_type_ = type;
}

std::string FileOutputNode::getOutputDir() const {
    return output_dir_;
}

void FileOutputNode::close() {
    if (output_) {
        output_->close();
        output_.reset();
    }
}

bool FileOutputNode::start() {
    if (running_)
        return true;

    // 创建输出配置
    pipeline::OutputConfig config;
    config.type = pipeline::OutputType::FILE;
    config.output_dir = output_dir_;

    // 创建输出
    output_ = pipeline::createOutput(config);
    if (!output_) {
        std::cerr << "[FileOutputNode] Failed to create output" << std::endl;
        return false;
    }

    running_ = true;
    std::cout << "[FileOutputNode] Started, output dir: " << output_dir_ << std::endl;
    return true;
}

void FileOutputNode::stop() {
    if (!running_)
        return;
    running_ = false;
    close();
    std::cout << "[FileOutputNode] Stopped" << std::endl;
}

void FileOutputNode::pushData(std::shared_ptr<core::BasePacket> packet) {
    auto det_packet = std::dynamic_pointer_cast<core::DetectionPacket>(packet);
    if (!det_packet) {
        std::cerr << "[FileOutputNode] Invalid packet type" << std::endl;
        return;
    }

    if (!output_) {
        std::cerr << "[FileOutputNode] Output not initialized" << std::endl;
        return;
    }

    // 转换检测结果
    std::vector<pipeline::Detection> detections;
    for (const auto& det : det_packet->detections) {
        pipeline::Detection detection;
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
        detections.push_back(detection);
    }

    // 写入文件
    output_->write(detections, det_packet->frame_id);
}

// 注册节点
REGISTER_NODE("file_output", FileOutputNode)

} // namespace nodes
} // namespace lidar_core
