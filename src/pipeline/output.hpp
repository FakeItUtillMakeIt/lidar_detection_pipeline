#ifndef __OUTPUT_HPP__
#define __OUTPUT_HPP__

#include <memory>
#include "types.hpp"

namespace pipeline {

class DetectionOutput {
public:
    virtual ~DetectionOutput() = default;
    virtual bool init(const OutputConfig& config) = 0;
    virtual void write(const std::vector<Detection>& dets, uint64_t frame_id) = 0;
    virtual void close() = 0;
    virtual OutputType type() const = 0;
};

std::shared_ptr<DetectionOutput> createOutput(const OutputConfig& config);

}  // namespace pipeline

#endif  // __OUTPUT_HPP__
