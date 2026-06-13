#include "callback_output.hpp"

namespace pipeline {

bool CallbackOutput::init(const OutputConfig& config) {
    return true;
}

void CallbackOutput::write(const std::vector<Detection>& dets, uint64_t frame_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (callback_) {
        callback_(dets, frame_id);
    }
}

void CallbackOutput::close() {}

void CallbackOutput::setCallback(DetectionCallback cb) {
    std::lock_guard<std::mutex> lock(mutex_);
    callback_ = std::move(cb);
}

}  // namespace pipeline
