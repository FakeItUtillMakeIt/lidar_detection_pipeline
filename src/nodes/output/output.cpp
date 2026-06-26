#include "output.hpp"
#include "file_writer.hpp"

namespace pipeline {

std::shared_ptr<DetectionOutput> createOutput(const OutputConfig& config) {
    switch (config.type) {
        case OutputType::FILE: {
            auto out = std::make_shared<FileWriter>();
            out->init(config);
            return out;
        }
        default:
            return nullptr;
    }
}

}  // namespace pipeline
