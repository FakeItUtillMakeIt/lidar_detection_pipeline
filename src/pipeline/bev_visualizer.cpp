#ifdef WITH_OPENCV

#include "bev_visualizer.hpp"
#include <iostream>

namespace pipeline {

bool BEVVisualizer::init(const OutputConfig& config) {
    cv::namedWindow(window_name_, cv::WINDOW_AUTOSIZE);
    return true;
}

void BEVVisualizer::setPointCloud(const PointCloud& cloud) {
    std::lock_guard<std::mutex> lock(mutex_);
    cloud_ = cloud;
    has_cloud_ = true;
}

void BEVVisualizer::write(const std::vector<Detection>& dets, uint64_t frame_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    detections_ = dets;
    last_frame_id_ = frame_id;
    renderBEV();
}

void BEVVisualizer::close() {
    cv::destroyAllWindows();
}

void BEVVisualizer::drawBox(cv::Mat& img, const Detection& det, const cv::Scalar& color) {
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

    // Label
    const char* labels[] = {"Car", "Ped", "Cyc"};
    const char* label = (det.class_id >= 0 && det.class_id < 3) ? labels[det.class_id] : "?";
    int px = static_cast<int>((det.x - origin_x_) * scale_);
    int py = static_cast<int>((det.y - origin_y_) * scale_) - 5;
    cv::putText(img, label, cv::Point(px, py), cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
}

void BEVVisualizer::renderBEV() {
    cv::Mat img = cv::Mat::zeros(height_, width_, CV_8UC3);

    // Draw point cloud
    if (has_cloud_) {
        for (const auto& p : cloud_.points) {
            int px = static_cast<int>((p.x - origin_x_) * scale_);
            int py = static_cast<int>((p.y - origin_y_) * scale_);
            if (px >= 0 && px < width_ && py >= 0 && py < height_) {
                uint8_t intensity = static_cast<uint8_t>(p.intensity * 255);
                img.at<cv::Vec3b>(py, px) = {intensity, intensity, intensity};
            }
        }
    }

    // Draw detections
    cv::Scalar colors[] = {
        {0, 255, 0},    // Car = green
        {0, 0, 255},    // Pedestrian = red
        {255, 0, 0},    // Cyclist = blue
    };

    for (const auto& det : detections_) {
        int cls = (det.class_id >= 0 && det.class_id < 3) ? det.class_id : 0;
        drawBox(img, det, colors[cls]);
    }

    // Info text
    char buf[128];
    snprintf(buf, sizeof(buf), "Frame: %lu | Dets: %zu",
             last_frame_id_, detections_.size());
    cv::putText(img, buf, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, {255, 255, 255}, 1);

    cv::imshow(window_name_, img);
    cv::waitKey(1);
}

}  // namespace pipeline

#endif  // WITH_OPENCV
