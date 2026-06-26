// src/nodes/source/bin_source_node.cpp
#include "3rd_party/log_mgr/log_mgr.h"
#include "bin_source_node.h"
#include "node_factory.h"
#include <iostream>

namespace lidar_core {
namespace nodes {

BinSourceNode::BinSourceNode() : ISourceNode("BinSourceNode") {
}

BinSourceNode::~BinSourceNode() {
    stop();
}

void BinSourceNode::setInputPath(const std::string& path) {
    input_path_ = path;
}

void BinSourceNode::setInputType(const std::string& type) {
    input_type_ = type;
}

std::string BinSourceNode::getInputPath() const {
    return input_path_;
}

bool BinSourceNode::start() {
    if (running_)
        return true;

    // 创建读取器配置
    pipeline::ReaderConfig config;
    config.input_path = input_path_;
    
    if (input_type_ == "velodyne") {
        config.type = pipeline::ReaderType::VELODYNE_UDP;
    } else {
        config.type = pipeline::ReaderType::BIN_FILE;
    }

    // 创建读取器
    reader_ = pipeline::createReader(config);
    if (!reader_ || !reader_->open()) {
        LOG_ERROR_FMT("[BinSourceNode] Failed to open input: {}", input_path_);
        return false;
    }

    running_ = true;
    LOG_INFO_FMT("[BinSourceNode] Started, input: {}", input_path_);
    return true;
}

void BinSourceNode::stop() {
    if (!running_)
        return;
    running_ = false;
    
    if (reader_) {
        reader_->close();
    }
    LOG_INFO_FMT("[BinSourceNode] Stopped");
}

void BinSourceNode::pushData(std::shared_ptr<core::BasePacket> packet) {
    // 源节点不接收外部数据，由外部调用readNext
}

bool BinSourceNode::readNext(std::shared_ptr<core::PointCloudPacket> packet) {
    if (!reader_ || !packet)
        return false;

    pipeline::PointCloud cloud;
    if (!reader_->read(cloud))
        return false;

    // 转换数据
    packet->frame_id = cloud.frame_id;
    packet->timestamp_ns = cloud.timestamp_ns;
    packet->points.reserve(cloud.points.size());
    
    for (const auto& pt : cloud.points) {
        core::PointXYZI point;
        point.x = pt.x;
        point.y = pt.y;
        point.z = pt.z;
        point.intensity = pt.intensity;
        packet->points.push_back(point);
    }

    return true;
}

// 注册节点
REGISTER_NODE("bin_source", BinSourceNode)

} // namespace nodes
} // namespace lidar_core
