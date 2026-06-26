// src/nodes/output/bev_visualizer_node.h
#pragma once

#include "lidar_core/nodes/i_output_node.h"
#include <memory>
#include <mutex>

#ifdef WITH_OPENCV
#include <opencv2/opencv.hpp>
#endif

namespace lidar_core {
namespace nodes {

class BEVVisualizerNode : public IOutputNode {
public:
    BEVVisualizerNode();
    ~BEVVisualizerNode() override;

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
#ifdef WITH_OPENCV
    void renderBEV(const std::vector<core::PointXYZI>& points,
                   const std::vector<core::Detection>& detections,
                   uint64_t frame_id);
    void drawBox(cv::Mat& img, const core::Detection& det, const cv::Scalar& color);
#endif

    std::string output_dir_;
    std::string window_name_ = "BEV Detection";
    bool show_window_ = false;  // 是否显示窗口
    double scale_ = 10.0;
    int width_ = 1400;
    int height_ = 800;
    float origin_x_ = 0.0;
    float origin_y_ = -39.68;
    std::mutex mutex_;
};

} // namespace nodes
} // namespace lidar_core
