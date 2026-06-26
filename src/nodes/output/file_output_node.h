// src/nodes/output/file_output_node.h
#pragma once

#include "lidar_core/nodes/i_output_node.h"
#include "output.hpp"
#include <memory>

namespace lidar_core {
namespace nodes {

class FileOutputNode : public IOutputNode {
public:
    FileOutputNode();
    ~FileOutputNode() override;

    // IOutputNode 接口
    void setOutputDir(const std::string& dir) override;
    void setOutputType(const std::string& type) override;
    std::string getOutputDir() const override;
    void close() override;

    // Node 接口
    bool start() override;
    void stop() override;
    void pushData(std::shared_ptr<core::BasePacket> packet) override;

private:
    std::string output_dir_;
    std::string output_type_;
    std::shared_ptr<pipeline::DetectionOutput> output_;
};

} // namespace nodes
} // namespace lidar_core
