// src/nodes/track/ocsort_adapter.cpp
#include "ocsort_adapter.h"
#include "OCSort.hpp"
#include <Eigen/Dense>
#include <iostream>

namespace lidar_core {
namespace nodes {

struct OCSortAdapter::Impl {
    std::unique_ptr<ocsort::OCSort> tracker;
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

void OCSortAdapter::update(std::vector<core::Detection>& detections) {
    if (!impl_->tracker || detections.empty()) {
        return;
    }

    int n = detections.size();

    // 中心格式 → 角点格式: [x1, y1, x2, y2, score, class_id]
    Eigen::MatrixXf dets(n, 6);
    for (int i = 0; i < n; i++) {
        float half_w = detections[i].w / 2.0f;
        float half_l = detections[i].l / 2.0f;
        dets(i, 0) = detections[i].x - half_w;  // x1
        dets(i, 1) = detections[i].y - half_l;  // y1
        dets(i, 2) = detections[i].x + half_w;  // x2
        dets(i, 3) = detections[i].y + half_l;  // y2
        dets(i, 4) = detections[i].score;
        dets(i, 5) = static_cast<float>(detections[i].class_id);
    }

    auto results = impl_->tracker->update(dets);

    // OCSort 返回的是原始检测的角点坐标，直接坐标匹配
    // 输出格式: [x1, y1, x2, y2, track_id, class_id, score]
    for (int i = 0; i < n; i++) {
        detections[i].track_id = -1;
    }

    for (const auto& row : results) {
        int track_id = static_cast<int>(row(4));
        float rx1 = row(0), ry1 = row(1), rx2 = row(2), ry2 = row(3);

        // 用中心坐标匹配（加容差）
        float cx = (rx1 + rx2) / 2.0f;
        float cy = (ry1 + ry2) / 2.0f;
        // 在原始检测中找到匹配的检测，赋值track_id
        for (int i = 0; i < n; i++) {
            if (detections[i].track_id >= 0) continue;
            if (std::abs(detections[i].x - cx) < 0.01f &&
                std::abs(detections[i].y - cy) < 0.01f) {
                detections[i].track_id = track_id;
                break;
            }
        }
    }
}

int OCSortAdapter::getActiveTrackCount() const {
    if (impl_->tracker) {
        return impl_->tracker->trackers.size();
    }
    return 0;
}

void OCSortAdapter::reset() {
    impl_->tracker.reset();
}

} // namespace nodes
} // namespace lidar_core
