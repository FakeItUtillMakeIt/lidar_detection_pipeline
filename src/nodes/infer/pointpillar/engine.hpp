#ifndef __ENGINE_HPP__
#define __ENGINE_HPP__

#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <cuda_runtime_api.h>
#include "lidar_core/core/types.hpp"
#include "detector.hpp"

namespace pipeline {

class PointPillarsEngine {
public:
    explicit PointPillarsEngine(const EngineConfig& config);
    ~PointPillarsEngine();

    bool init();

    // Sync mode
    std::vector<Detection> detect(const PointCloud& cloud);

    // Async mode
    void detectAsync(PointCloud cloud, DetectionCallback callback);
    void detectAsyncWithVis(PointCloud cloud, DetectionCallback callback,
                            std::function<void(const PointCloud&)> vis_callback);

    void shutdown();

private:
    struct InferenceTask {
        PointCloud cloud;
        DetectionCallback callback;
        std::function<void(const PointCloud&)> vis_callback;
    };

    void workerLoop();
    std::vector<Detection> toDetections(const std::vector<pointpillar::lidar::BoundingBox>& bboxes, uint64_t frame_id);

    EngineConfig config_;
    std::shared_ptr<pointpillar::lidar::Detector> detector_;
    cudaStream_t stream_ = nullptr;

    // Thread pool
    std::vector<std::thread> workers_;
    std::queue<InferenceTask> task_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    bool shutdown_ = false;
};

}  // namespace pipeline

#endif  // __ENGINE_HPP__
