// src/nodes/track/ocsort_adapter.cpp
#include "ocsort_adapter.h"
#include "3rd_party/tracker/OCSort/OCSort.hpp"
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

    // OCSort 期望输入格式: [x1, y1, x2, y2, score, class_id] (角点格式)
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

    // OCSort 输出格式: [x1, y1, x2, y2, track_id, class_id, score]
    // 用 track_id 标注原始检测，通过 IoU 匹配
    for (int i = 0; i < n; i++) {
        detections[i].track_id = -1;
    }

    for (const auto& row : results) {
        int track_id = static_cast<int>(row(4));
        int cls = static_cast<int>(row(5));
        float rx1 = row(0), ry1 = row(1), rx2 = row(2), ry2 = row(3);

        // 找 IoU 最大的原始检测进行匹配
        int best_idx = -1;
        float best_iou = 0.3f;

        for (int i = 0; i < n; i++) {
            if (detections[i].track_id >= 0) continue; // 已匹配

            float half_w = detections[i].w / 2.0f;
            float half_l = detections[i].l / 2.0f;
            float ox1 = detections[i].x - half_w;
            float oy1 = detections[i].y - half_l;
            float ox2 = detections[i].x + half_w;
            float oy2 = detections[i].y + half_l;

            // 计算 IoU
            float inter_x1 = std::max(ox1, rx1);
            float inter_y1 = std::max(oy1, ry1);
            float inter_x2 = std::min(ox2, rx2);
            float inter_y2 = std::min(oy2, ry2);
            float inter = std::max(0.0f, inter_x2 - inter_x1) * std::max(0.0f, inter_y2 - inter_y1);
            float area_o = (ox2 - ox1) * (oy2 - oy1);
            float area_r = (rx2 - rx1) * (ry2 - ry1);
            float iou = inter / (area_o + area_r - inter + 1e-6f);

            if (iou > best_iou) {
                best_iou = iou;
                best_idx = i;
            }
        }

        if (best_idx >= 0) {
            detections[best_idx].track_id = track_id;
            detections[best_idx].class_id = cls;
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
