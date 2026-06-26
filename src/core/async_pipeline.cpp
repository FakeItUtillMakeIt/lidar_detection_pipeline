// src/core/async_pipeline.cpp
#include "lidar_core/core/async_pipeline.h"
#include "lidar_core/nodes/i_source_node.h"
#include "lidar_core/nodes/i_infer_node.h"
#include "nodes/source/bin_source_node.h"
#include <iostream>

namespace lidar_core {
namespace core {

AsyncPipeline::AsyncPipeline(const std::string& id, size_t queue_size)
    : id_(id), queue_size_(queue_size), input_queue_(queue_size) {
    pipeline_ = std::make_shared<Pipeline>(id);
}

AsyncPipeline::~AsyncPipeline() {
    stop();
}

bool AsyncPipeline::buildFromJson(const nlohmann::json& config) {
    return pipeline_->buildFromJson(config);
}

bool AsyncPipeline::start() {
    if (running_) return true;

    if (!pipeline_->start()) {
        return false;
    }

    running_ = true;
    start_time_ = std::chrono::high_resolution_clock::now();

    // 启动读取线程和推理线程
    reader_thread_ = std::thread(&AsyncPipeline::readerLoop, this);
    infer_thread_ = std::thread(&AsyncPipeline::inferLoop, this);

    std::cout << "[AsyncPipeline " << id_ << "] Started with queue_size=" << queue_size_ << std::endl;
    return true;
}

void AsyncPipeline::stop() {
    if (!running_) return;

    running_ = false;
    input_queue_.stop();

    if (reader_thread_.joinable()) reader_thread_.join();
    if (infer_thread_.joinable()) infer_thread_.join();

    pipeline_->stop();
    std::cout << "[AsyncPipeline " << id_ << "] Stopped" << std::endl;
}

bool AsyncPipeline::isRunning() const {
    return running_ && !infer_finished_;
}

double AsyncPipeline::getFPS() const {
    auto now = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(now - start_time_).count();
    return sec > 0 ? processed_frames_.load() / sec : 0;
}

void AsyncPipeline::readerLoop() {
    auto source_node = std::dynamic_pointer_cast<nodes::ISourceNode>(
        pipeline_->getNode("source"));
    auto bin_source = std::dynamic_pointer_cast<nodes::BinSourceNode>(source_node);

    if (!bin_source) {
        std::cerr << "[AsyncPipeline] source node not found or wrong type" << std::endl;
        running_ = false;
        return;
    }

    std::cout << "[AsyncPipeline] Reader thread started" << std::endl;

    while (running_) {
        auto cloud = std::make_shared<PointCloudPacket>();
        if (!bin_source->readNext(cloud)) {
            break;  // 数据读完
        }

        // 非阻塞推入队列
        while (running_ && !input_queue_.try_push(cloud)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    // 通知推理线程没有更多数据
    input_queue_.stop();
    std::cout << "[AsyncPipeline] Reader thread finished" << std::endl;
}

void AsyncPipeline::inferLoop() {
    auto infer_node = std::dynamic_pointer_cast<nodes::IInferNode>(
        pipeline_->getNode("infer"));

    if (!infer_node) {
        std::cerr << "[AsyncPipeline] infer node not found" << std::endl;
        running_ = false;
        return;
    }

    std::cout << "[AsyncPipeline] Infer thread started" << std::endl;

    while (running_) {
        auto cloud = input_queue_.pop();
        if (!cloud) break;  // 队列关闭

        // 执行推理（会自动 broadcast 到下游节点）
        infer_node->pushData(*cloud);

        processed_frames_++;
    }

    infer_finished_ = true;
    std::cout << "[AsyncPipeline] Infer thread finished" << std::endl;
}

} // namespace core
} // namespace lidar_core
