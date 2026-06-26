// include/lidar_core/core/async_pipeline.h
#pragma once

#include "pipeline.h"
#include "thread_safe_queue.h"
#include <thread>
#include <atomic>
#include <functional>

namespace lidar_core {
namespace core {

// 异步管道：将读取和推理解耦，提高吞吐
// 读取线程 -> 队列 -> 推理线程 -> 下游处理
class AsyncPipeline {
public:
    explicit AsyncPipeline(const std::string& id, size_t queue_size = 3);
    ~AsyncPipeline();

    // 从 JSON 构建管道
    bool buildFromJson(const nlohmann::json& config);

    // 启动异步处理
    bool start();

    // 停止
    void stop();

    // 是否运行中
    bool isRunning() const;

    // 获取底层管道（用于访问节点）
    std::shared_ptr<Pipeline> getPipeline() const { return pipeline_; }

    // 获取统计信息
    uint64_t getProcessedFrames() const { return processed_frames_.load(); }
    double getFPS() const;

private:
    // 读取线程函数
    void readerLoop();

    // 推理线程函数
    void inferLoop();

    std::string id_;
    std::shared_ptr<Pipeline> pipeline_;
    size_t queue_size_;

    // 线程
    std::thread reader_thread_;
    std::thread infer_thread_;
    std::atomic<bool> running_{false};

    // 队列
    using CloudQueue = ThreadSafeQueue<std::shared_ptr<core::PointCloudPacket>>;
    CloudQueue input_queue_;

    // 统计
    std::atomic<uint64_t> processed_frames_{0};
    std::atomic<bool> infer_finished_{false};
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
};

} // namespace core
} // namespace lidar_core
