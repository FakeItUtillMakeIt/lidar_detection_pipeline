// src/nodes/output/bev_visualizer_node.cpp
#include "bev_visualizer_node.h"
#include "node_factory.h"
#include <iostream>
#include <sys/stat.h>

namespace lidar_core {
namespace nodes {

BEVVisualizerNode::BEVVisualizerNode() : IOutputNode("BEVVisualizerNode") {
}

BEVVisualizerNode::~BEVVisualizerNode() {
    close();
}

void BEVVisualizerNode::setOutputDir(const std::string& dir) {
    output_dir_ = dir;
}

void BEVVisualizerNode::setOutputType(const std::string& type) {
    // check for DISPLAY environment variable
    const char* display = getenv("DISPLAY");
    show_window_ = (display != nullptr && display[0] != '\0');
}

std::string BEVVisualizerNode::getOutputDir() const {
    return output_dir_;
}

void BEVVisualizerNode::close() {
#ifdef WITH_OPENCV
    if (show_window_) {
        cv::destroyAllWindows();
    }
#endif
}

bool BEVVisualizerNode::start() {
    if (running_)
        return true;

#ifdef WITH_OPENCV
    // 创建输出目录
    if (!output_dir_.empty()) {
        mkdir(output_dir_.c_str(), 0755);
    }

    if (show_window_) {
        cv::namedWindow(window_name_, cv::WINDOW_AUTOSIZE);
    }
#else
    std::cerr << "[BEVVisualizerNode] OpenCV not available" << std::endl;
    return false;
#endif

    running_ = true;
    std::cout << "[BEVVisualizerNode] Started, output: " << output_dir_ << std::endl;
    return true;
}

void BEVVisualizerNode::stop() {
    if (!running_)
        return;
    running_ = false;
    close();
    std::cout << "[BEVVisualizerNode] Stopped" << std::endl;
}

void BEVVisualizerNode::pushData(std::shared_ptr<core::BasePacket> packet) {
    auto det_packet = std::dynamic_pointer_cast<core::DetectionPacket>(packet);
    if (!det_packet) {
        std::cerr << "[BEVVisualizerNode] Invalid packet type" << std::endl;
        return;
    }

#ifdef WITH_OPENCV
    renderBEV(det_packet->cloud_points, det_packet->detections, det_packet->frame_id);
#endif
}

#ifdef WITH_OPENCV

void BEVVisualizerNode::renderBEV(const std::vector<core::PointXYZI>& points,
                                   const std::vector<core::Detection>& detections,
                                   uint64_t frame_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    cv::Mat img = cv::Mat::zeros(height_, width_, CV_8UC3);

    // 绘制点云
    for (const auto& p : points) {
        int px = static_cast<int>((p.x - origin_x_) * scale_);
        int py = static_cast<int>((p.y - origin_y_) * scale_);
        if (px >= 0 && px < width_ && py >= 0 && py < height_) {
            uint8_t intensity = static_cast<uint8_t>(p.intensity * 255);
            img.at<cv::Vec3b>(py, px) = {intensity, intensity, intensity};
        }
    }

    // 绘制检测框
    cv::Scalar colors[] = {
        {0, 255, 0},    // Car = green
        {0, 0, 255},    // Pedestrian = red
        {255, 0, 0},    // Cyclist = blue
    };

    for (const auto& det : detections) {
        int cls = (det.class_id >= 0 && det.class_id < 3) ? det.class_id : 0;
        drawBox(img, det, colors[cls]);
    }

    // 显示信息
    char buf[128];
    snprintf(buf, sizeof(buf), "Frame: %lu | Dets: %zu",
             frame_id, detections.size());
    cv::putText(img, buf, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, {255, 255, 255}, 1);

    // 显示窗口或保存图片
    if (show_window_) {
        cv::imshow(window_name_, img);
        cv::waitKey(1);
    } else if (!output_dir_.empty()) {
        char filename[256];
        snprintf(filename, sizeof(filename), "%s/bev_%06lu.png", output_dir_.c_str(), frame_id);
        cv::imwrite(filename, img);
    }
}

void BEVVisualizerNode::drawBox(cv::Mat& img, const core::Detection& det, const cv::Scalar& color) {
    float cos_a = cos(det.rt);
    float sin_a = sin(det.rt);
    float half_l = det.l / 2;
    float half_w = det.w / 2;

    cv::Point2f corners[4] = {
        {det.x - half_l * cos_a - half_w * sin_a, det.y - half_l * sin_a + half_w * cos_a},
        {det.x + half_l * cos_a - half_w * sin_a, det.y + half_l * sin_a + half_w * cos_a},
        {det.x + half_l * cos_a + half_w * sin_a, det.y + half_l * sin_a - half_w * cos_a},
        {det.x - half_l * cos_a + half_w * sin_a, det.y - half_l * sin_a - half_w * cos_a},
    };

    std::vector<cv::Point> pts;
    for (int i = 0; i < 4; i++) {
        int px = static_cast<int>((corners[i].x - origin_x_) * scale_);
        int py = static_cast<int>((corners[i].y - origin_y_) * scale_);
        pts.emplace_back(px, py);
    }

    cv::polylines(img, pts, true, color, 2, cv::LINE_AA);

    // 标签
    const char* labels[] = {"Car", "Ped", "Cyc"};
    const char* label = (det.class_id >= 0 && det.class_id < 3) ? labels[det.class_id] : "?";
    int px = static_cast<int>((det.x - origin_x_) * scale_);
    int py = static_cast<int>((det.y - origin_y_) * scale_) - 5;
    cv::putText(img, label, cv::Point(px, py), cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
}

#endif  // WITH_OPENCV

// 注册节点
REGISTER_NODE("bev_visualizer", BEVVisualizerNode)

} // namespace nodes
} // namespace lidar_core
