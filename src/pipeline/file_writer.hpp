#ifndef __FILE_WRITER_HPP__
#define __FILE_WRITER_HPP__

#include "output.hpp"
#include <string>

namespace pipeline {

class FileWriter : public DetectionOutput {
public:
    bool init(const OutputConfig& config) override;
    void write(const std::vector<Detection>& dets, uint64_t frame_id) override;
    void close() override;
    OutputType type() const override { return OutputType::FILE; }

private:
    std::string output_dir_;
};

}  // namespace pipeline

#endif  // __FILE_WRITER_HPP__
