#include "3rd_party/log_mgr/log_mgr.h"
#include "engine.hpp"
#include <iostream>
#include <cstring>

namespace pipeline {

PointPillarsEngine::PointPillarsEngine(const EngineConfig& config) : config_(config) {}

PointPillarsEngine::~PointPillarsEngine() { shutdown(); }

bool PointPillarsEngine::init() {
    pointpillar::lidar::DetectorConfig det_config;
    memcpy(det_config.min_range, config_.min_range, sizeof(float) * 3);
    memcpy(det_config.max_range, config_.max_range, sizeof(float) * 3);
    memcpy(det_config.voxel_size, config_.voxel_size, sizeof(float) * 3);
    det_config.max_voxels = config_.max_voxels;
    det_config.max_points_per_voxel = config_.max_points_per_voxel;
    det_config.max_points = config_.max_points;
    det_config.model_path = config_.model_path;
    det_config.score_thresh = config_.score_thresh;
    det_config.nms_thresh = config_.nms_thresh;

    detector_ = pointpillar::lidar::Detector::create(det_config);
    if (!detector_) {
        LOG_ERROR_FMT("[PointPillarsEngine] Failed to create detector");
        return false;
    }

    cudaStreamCreate(&stream_);
    detector_->print();

    // Start worker threads for async mode
    if (config_.async_mode) {
        for (int i = 0; i < config_.num_workers; i++) {
            workers_.emplace_back(&PointPillarsEngine::workerLoop, this);
        }
        LOG_INFO_FMT("[PointPillarsEngine] Async engine started with {} workers", config_.num_workers);
    }

    return true;
}

std::vector<Detection> PointPillarsEngine::detect(const PointCloud& cloud) {
    auto bboxes = detector_->detect(
        reinterpret_cast<const float*>(cloud.points.data()),
        static_cast<int>(cloud.points.size()),
        stream_
    );
    LOG_INFO_FMT("[PointPillarsEngine] Frame {}: Detected {} objects", cloud.frame_id, bboxes.size());
    return toDetections(bboxes, cloud.frame_id);
}

void PointPillarsEngine::detectAsync(PointCloud cloud, DetectionCallback callback) {
    detectAsyncWithVis(std::move(cloud), std::move(callback), nullptr);
}

void PointPillarsEngine::detectAsyncWithVis(PointCloud cloud, DetectionCallback callback,
                                            std::function<void(const PointCloud&)> vis_callback) {
    InferenceTask task;
    task.cloud = std::move(cloud);
    task.callback = std::move(callback);
    task.vis_callback = std::move(vis_callback);

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        task_queue_.push(std::move(task));
    }
    queue_cv_.notify_one();
}

void PointPillarsEngine::workerLoop() {
    while (true) {
        InferenceTask task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] { return shutdown_ || !task_queue_.empty(); });
            if (shutdown_ && task_queue_.empty()) return;
            task = std::move(task_queue_.front());
            task_queue_.pop();
        }

        // Run inference
        auto bboxes = detector_->detect(
            reinterpret_cast<const float*>(task.cloud.points.data()),
            static_cast<int>(task.cloud.points.size()),
            stream_
        );
        auto detections = toDetections(bboxes, task.cloud.frame_id);
        LOG_INFO_FMT("[PointPillarsEngine] Frame {}: Detected {} objects", task.cloud.frame_id, detections.size());
        // Callback
        if (task.callback) {
            task.callback(detections, task.cloud.frame_id);
        }

        // Visualization callback
        if (task.vis_callback) {
            task.vis_callback(task.cloud);
        }
    }
}

std::vector<Detection> PointPillarsEngine::toDetections(
    const std::vector<pointpillar::lidar::BoundingBox>& bboxes, uint64_t frame_id)
{
    std::vector<Detection> result;
    result.reserve(bboxes.size());
    for (const auto& bb : bboxes) {
        Detection det;
        det.x = bb.x;
        det.y = bb.y;
        det.z = bb.z;
        det.w = bb.w;
        det.l = bb.l;
        det.h = bb.h;
        det.rt = bb.rt;
        det.class_id = bb.id;
        det.score = bb.score;
        det.frame_id = frame_id;
        result.push_back(det);
    }
    return result;
}

void PointPillarsEngine::shutdown() {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        shutdown_ = true;
    }
    queue_cv_.notify_all();
    for (auto& t : workers_) {
        if (t.joinable()) t.join();
    }
    workers_.clear();

    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

}  // namespace pipeline
