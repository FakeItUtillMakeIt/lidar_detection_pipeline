#include "output.hpp"
#include "file_writer.hpp"

#include <cstdio>

namespace pipeline {

std::shared_ptr<DetectionOutput> createOutput(const OutputConfig& config) {
    switch (config.type) {
        case OutputType::FILE: {
            auto out = std::make_shared<FileWriter>();
            out->init(config);
            return out;
        }
        case OutputType::CALLBACK:
        case OutputType::ROS2:
        case OutputType::BEV_VISUALIZER:
            std::fprintf(stderr, "[Output] OutputType %d not yet implemented\n",
                         static_cast<int>(config.type));
            return nullptr;
        default:
            std::fprintf(stderr, "[Output] Unknown OutputType %d\n",
                         static_cast<int>(config.type));
            return nullptr;
    }
}

}  // namespace pipeline
