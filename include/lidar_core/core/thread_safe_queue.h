// include/lidar_core/core/thread_safe_queue.h
#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>

namespace lidar_core {
namespace core {

template<typename T>
class ThreadSafeQueue {
public:
    explicit ThreadSafeQueue(size_t max_size = 10) : max_size_(max_size) {}

    // 阻塞推入（队列满时等待）
    void push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_.wait(lock, [this] { return queue_.size() < max_size_ || stopped_; });
        if (stopped_) return;
        queue_.push(std::move(item));
        not_empty_.notify_one();
    }

    // 非阻塞推入（队列满时返回false）
    bool try_push(T item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.size() >= max_size_ || stopped_) return false;
        queue_.push(std::move(item));
        not_empty_.notify_one();
        return true;
    }

    // 阻塞弹出（队列空时等待）
    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [this] { return !queue_.empty() || stopped_; });
        if (stopped_ && queue_.empty()) return std::nullopt;
        T item = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return item;
    }

    // 非阻塞弹出
    std::optional<T> try_pop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return std::nullopt;
        T item = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return item;
    }

    void stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::queue<T> queue_;
    size_t max_size_;
    bool stopped_ = false;
};

} // namespace core
} // namespace lidar_core
