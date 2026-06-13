#ifndef __CALLBACK_OUTPUT_HPP__
#define __CALLBACK_OUTPUT_HPP__

#include "output.hpp"
#include <mutex>

namespace pipeline {

class CallbackOutput : public DetectionOutput {
public:
    bool init(const OutputConfig& config) override;
    void write(const std::vector<Detection>& dets, uint64_t frame_id) override;
    void close() override;
    OutputType type() const override { return OutputType::CALLBACK; }

    void setCallback(DetectionCallback cb);

private:
    DetectionCallback callback_;
    std::mutex mutex_;
};

}  // namespace pipeline

#endif  // __CALLBACK_OUTPUT_HPP__
