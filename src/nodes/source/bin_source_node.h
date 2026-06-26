// src/nodes/source/bin_source_node.h
#pragma once

#include "lidar_core/nodes/i_source_node.h"
#include "reader.hpp"
#include <memory>

namespace lidar_core {
namespace nodes {

class BinSourceNode : public ISourceNode {
public:
    BinSourceNode();
    ~BinSourceNode() override;

    // ISourceNode 接口
    void setInputPath(const std::string& path) override;
    void setInputType(const std::string& type) override;
    std::string getInputPath() const override;

    // Node 接口
    bool start() override;
    void stop() override;
    void pushData(std::shared_ptr<core::BasePacket> packet) override;

    // 读取下一帧数据
    bool readNext(std::shared_ptr<core::PointCloudPacket> packet);

private:
    std::string input_path_;
    std::string input_type_;
    std::shared_ptr<pipeline::PointCloudReader> reader_;
};

} // namespace nodes
} // namespace lidar_core
