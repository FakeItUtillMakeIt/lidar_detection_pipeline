// src/nodes/track/ocsort_adapter.cpp
#include "ocsort_adapter.h"
#include "3rd_party/tracker/OCSort/OCSort.hpp"
#include <Eigen/Dense>
#include <iostream>

namespace lidar_core {
namespace nodes {

struct OCSortAdapter::Impl {
    std::unique_ptr<ocsort::OCSort> tracker;
    int next_id = 0;
    std::unordered_map<int, core::Detection> prev_detections;
};

OCSortAdapter::OCSortAdapter() : impl_(std::make_unique<Impl>()) {}

OCSortAdapter::~OCSortAdapter() = default;

bool OCSortAdapter::init(float det_thresh, int max_age, int min_hits,
                          float iou_threshold, int delta_t, float inertia, bool use_byte) {
    impl_->tracker = std::make_unique<ocsort::OCSort>(
        det_thresh, max_age, min_hits, iou_threshold,
        delta_t, "iou", inertia, use_byte
    );
    return true;
}

std::vector<core::Detection> OCSortAdapter::update(const std::vector<core::Detection>& detections) {
    if (!impl_->tracker) {
        return detections;
    }

    // 转换为OCSort格式 [x, y, w, h, score, class_id]
    // 对于3D LiDAR，使用BEV投影: x, y为中心，w=width, h=length
    int n = detections.size();
    Eigen::MatrixXf dets(n, 6);
    
    for (int i = 0; i < n; i++) {
        dets(i, 0) = detections[i].x;  // center x
        dets(i, 1) = detections[i].y;  // center y
        dets(i, 2) = detections[i].w;  // width
        dets(i, 3) = detections[i].l;  // length (作为height维度)
        dets(i, 4) = detections[i].score;
        dets(i, 5) = static_cast<float>(detections[i].class_id);
    }

    // 运行OCSort
    auto results = impl_->tracker->update(dets);

    // 转换回Detection格式
    std::vector<core::Detection> tracked_detections;
    tracked_detections.reserve(results.size());

    for (const auto& row : results) {
        core::Detection det;
        det.x = row(0);           // center x
        det.y = row(1);           // center y
        det.w = row(2);           // width
        det.l = row(3);           // length
        det.track_id = static_cast<int>(row(4));  // track id
        det.class_id = static_cast<int>(row(5));  // class_id
        
        // 查找原始检测结果，获取z, h, rt, score等信息
        // OCSort的输出只包含BEV信息，需要从原始检测中补充3D信息
        for (const auto& orig : detections) {
            if (std::abs(orig.x - det.x) < 0.01f && 
                std::abs(orig.y - det.y) < 0.01f &&
                orig.class_id == det.class_id) {
                det.z = orig.z;
                det.h = orig.h;
                det.rt = orig.rt;
                det.score = orig.score;
                det.frame_id = orig.frame_id;
                break;
            }
        }

        // 从跟踪器获取速度信息
        // 注意：这里简化处理，实际应该从Kalman滤波器获取
        det.vx = 0;
        det.vy = 0;
        det.track_age = 0;
        det.track_active = (det.track_id >= 0);

        tracked_detections.push_back(det);
    }

    return tracked_detections;
}

int OCSortAdapter::getActiveTrackCount() const {
    if (impl_->tracker) {
        return impl_->tracker->trackers.size();
    }
    return 0;
}

void OCSortAdapter::reset() {
    impl_->tracker.reset();
    impl_->prev_detections.clear();
}

} // namespace nodes
} // namespace lidar_core
