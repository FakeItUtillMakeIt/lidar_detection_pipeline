#ifndef __BEV_VISUALIZER_HPP__
#define __BEV_VISUALIZER_HPP__

#ifdef WITH_OPENCV

#include "output.hpp"
#include <opencv2/opencv.hpp>
#include <mutex>

namespace pipeline {

class BEVVisualizer : public DetectionOutput {
public:
    bool init(const OutputConfig& config) override;
    void write(const std::vector<Detection>& dets, uint64_t frame_id) override;
    void close() override;
    OutputType type() const override { return OutputType::BEV_VISUALIZER; }

    void setPointCloud(const PointCloud& cloud);

private:
    void renderBEV();
    void drawBox(cv::Mat& img, const Detection& det, const cv::Scalar& color);

    PointCloud cloud_;
    std::vector<Detection> detections_;
    std::string window_name_ = "BEV Detection";
    double scale_ = 10.0;   // pixels per meter
    int width_ = 1400;      // canvas width
    int height_ = 800;      // canvas height
    float origin_x_ = 0.0;  // min x
    float origin_y_ = -39.68; // min y
    bool has_cloud_ = false;
    std::mutex mutex_;
    uint64_t last_frame_id_ = 0;
    double fps_ = 0;
};

}  // namespace pipeline

#endif  // WITH_OPENCV
#endif  // __BEV_VISUALIZER_HPP__
