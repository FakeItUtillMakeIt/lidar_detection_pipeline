// src/nodes/output/trajectory_visualizer_node.cpp
#include "3rd_party/log_mgr/log_mgr.h"
#include "trajectory_visualizer_node.h"
#include "node_factory.h"
#include <iostream>
#include <sys/stat.h>

namespace lidar_core {
namespace nodes {

TrajectoryVisualizerNode::TrajectoryVisualizerNode() : IOutputNode("TrajectoryVisualizerNode") {
}

TrajectoryVisualizerNode::~TrajectoryVisualizerNode() {
    close();
}

void TrajectoryVisualizerNode::setOutputDir(const std::string& dir) {
    output_dir_ = dir;
}

void TrajectoryVisualizerNode::setOutputType(const std::string& type) {
    const char* display = getenv("DISPLAY");
    show_window_ = (display != nullptr && display[0] != '\0');
}

std::string TrajectoryVisualizerNode::getOutputDir() const {
    return output_dir_;
}

void TrajectoryVisualizerNode::close() {
#ifdef WITH_OPENCV
    if (show_window_) {
        cv::destroyAllWindows();
    }
#endif
}

bool TrajectoryVisualizerNode::start() {
    if (running_)
        return true;

#ifdef WITH_OPENCV
    if (!output_dir_.empty()) {
        mkdir("./out", 0755);
        mkdir(output_dir_.c_str(), 0755);
    }

    if (show_window_) {
        cv::namedWindow(window_name_, cv::WINDOW_AUTOSIZE);
    }
#else
    LOG_ERROR_FMT("[TrajectoryVisualizerNode] OpenCV not available");
    return false;
#endif

    running_ = true;
    LOG_INFO_FMT("[TrajectoryVisualizerNode] Started, output: {}", output_dir_);
    return true;
}

void TrajectoryVisualizerNode::stop() {
    if (!running_)
        return;
    running_ = false;
    close();
    LOG_INFO_FMT("[TrajectoryVisualizerNode] Stopped");
}

void TrajectoryVisualizerNode::pushData(std::shared_ptr<core::BasePacket> packet) {
    auto plan_packet = std::dynamic_pointer_cast<core::PlanningPacket>(packet);
    if (!plan_packet) {
        LOG_ERROR_FMT("[TrajectoryVisualizerNode] Invalid packet type");
        return;
    }

#ifdef WITH_OPENCV
    renderTrajectory(plan_packet->cloud_points, 
                    plan_packet->obstacles,
                    plan_packet->trajectory,
                    plan_packet->is_feasible,
                    plan_packet->frame_id);
#endif
}

#ifdef WITH_OPENCV

void TrajectoryVisualizerNode::renderTrajectory(
    const std::vector<core::PointXYZI>& points,
    const std::vector<core::Detection>& detections,
    const std::vector<core::PlanningPacket::PathPoint>& trajectory,
    bool is_feasible,
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

    // 绘制规划轨迹
    drawTrajectory(img, trajectory, is_feasible);

    // 绘制目标点
    int goal_px = static_cast<int>((100.0f - origin_x_) * scale_);
    int goal_py = static_cast<int>((0.0f - origin_y_) * scale_);
    cv::circle(img, cv::Point(goal_px, goal_py), 10, {0, 255, 255}, -1);
    cv::putText(img, "Goal", cv::Point(goal_px + 15, goal_py), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 255, 255}, 1);

    // 显示信息
    char buf[128];
    snprintf(buf, sizeof(buf), "Frame: %lu | Dets: %zu | Traj: %zu | Feasible: %s",
             frame_id, detections.size(), trajectory.size(),
             is_feasible ? "Yes" : "No");
    cv::putText(img, buf, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, {255, 255, 255}, 1);

    // 显示规划状态
    char status_buf[64];
    snprintf(status_buf, sizeof(status_buf), "Status: %s", 
             is_feasible ? "Feasible" : "Infeasible");
    cv::putText(img, status_buf, cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.6, 
                is_feasible ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 1);

    // 显示窗口或保存图片
    if (show_window_) {
        cv::imshow(window_name_, img);
        cv::waitKey(1);
    } else if (!output_dir_.empty()) {
        char filename[256];
        snprintf(filename, sizeof(filename), "%s/trajectory_%06lu.png", 
                 output_dir_.c_str(), frame_id);
        cv::imwrite(filename, img);
    }
}

void TrajectoryVisualizerNode::drawBox(cv::Mat& img, const core::Detection& det, const cv::Scalar& color) {
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
    
    char text[64];
    if (det.track_id >= 0) {
        snprintf(text, sizeof(text), "%s#%d %.1fm/s", label, det.track_id, det.speed);
    } else {
        snprintf(text, sizeof(text), "%s", label);
    }
    cv::putText(img, text, cv::Point(px, py), cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
}

void TrajectoryVisualizerNode::drawTrajectory(
    cv::Mat& img, 
    const std::vector<core::PlanningPacket::PathPoint>& trajectory,
    bool is_feasible) {
    
    if (trajectory.empty()) return;

    // 根据可行性选择颜色
    cv::Scalar traj_color = is_feasible ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 128, 255);

    // 绘制轨迹点
    for (size_t i = 0; i < trajectory.size(); ++i) {
        int px = static_cast<int>((trajectory[i].x - origin_x_) * scale_);
        int py = static_cast<int>((trajectory[i].y - origin_y_) * scale_);
        
        // 绘制点
        cv::circle(img, cv::Point(px, py), 3, traj_color, -1);
        
        // 绘制连接线
        if (i > 0) {
            int prev_px = static_cast<int>((trajectory[i-1].x - origin_x_) * scale_);
            int prev_py = static_cast<int>((trajectory[i-1].y - origin_y_) * scale_);
            cv::line(img, cv::Point(prev_px, prev_py), cv::Point(px, py), traj_color, 2);
        }
    }

    // 绘制起点（自车位置）
    int start_px = static_cast<int>((0.0f - origin_x_) * scale_);
    int start_py = static_cast<int>((0.0f - origin_y_) * scale_);
    cv::circle(img, cv::Point(start_px, start_py), 8, {255, 255, 255}, -1);
    cv::putText(img, "Ego", cv::Point(start_px + 10, start_py), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, {255, 255, 255}, 1);
}

#endif  // WITH_OPENCV

// 注册节点
REGISTER_NODE("trajectory_visualizer", TrajectoryVisualizerNode)

} // namespace nodes
} // namespace lidar_core