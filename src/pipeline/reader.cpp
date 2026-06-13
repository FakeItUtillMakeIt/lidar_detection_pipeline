#include "reader.hpp"
#include "bin_reader.hpp"
#include "velodyne_reader.hpp"

namespace pipeline {

std::shared_ptr<PointCloudReader> createReader(const ReaderConfig& config) {
    switch (config.type) {
        case ReaderType::BIN_FILE:
            return std::make_shared<BinFileReader>(config);
        case ReaderType::VELODYNE_UDP:
            return std::make_shared<VelodyneReader>(config);
        default:
            return nullptr;
    }
}

}  // namespace pipeline
