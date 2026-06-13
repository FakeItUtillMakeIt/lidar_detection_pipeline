#include "output.hpp"
#include "file_writer.hpp"
#include "callback_output.hpp"
#ifdef WITH_OPENCV
#include "bev_visualizer.hpp"
#endif

namespace pipeline {

std::shared_ptr<DetectionOutput> createOutput(const OutputConfig& config) {
    switch (config.type) {
        case OutputType::FILE: {
            auto out = std::make_shared<FileWriter>();
            out->init(config);
            return out;
        }
        case OutputType::CALLBACK: {
            auto out = std::make_shared<CallbackOutput>();
            out->init(config);
            return out;
        }
        case OutputType::BEV_VISUALIZER: {
#ifdef WITH_OPENCV
            auto out = std::make_shared<BEVVisualizer>();
            out->init(config);
            return out;
#else
            return nullptr;
#endif
        }
        default:
            return nullptr;
    }
}

}  // namespace pipeline
