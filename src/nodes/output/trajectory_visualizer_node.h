// src/nodes/output/trajectory_visualizer_node.h
#pragma once

#include "lidar_core/nodes/i_output_node.h"
#include <memory>
#include <mutex>

#ifdef WITH_OPENCV
#include <opencv2/opencv.hpp>
#endif

namespace lidar_core {
namespace nodes {

class TrajectoryVisualizerNode : public IOutputNode {
public:
    TrajectoryVisualizerNode();
    ~TrajectoryVisualizerNode() override;

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
    void renderTrajectory(const std::vector<core::PointXYZI>& points,
                         const std::vector<core::Detection>& detections,
                         const std::vector<core::PlanningPacket::PathPoint>& trajectory,
                         bool is_feasible,
                         uint64_t frame_id);
    void drawBox(cv::Mat& img, const core::Detection& det, const cv::Scalar& color);
    void drawTrajectory(cv::Mat& img, 
                       const std::vector<core::PlanningPacket::PathPoint>& trajectory,
                       bool is_feasible);
#endif

    std::string output_dir_;
    std::string window_name_ = "Trajectory Planning";
    bool show_window_ = false;
    double scale_ = 10.0;
    int width_ = 1400;
    int height_ = 800;
    float origin_x_ = 0.0;
    float origin_y_ = -39.68;
    std::mutex mutex_;
};

} // namespace nodes
} // namespace lidar_core