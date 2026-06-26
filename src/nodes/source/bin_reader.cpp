#include "3rd_party/log_mgr/log_mgr.h"
#include "bin_reader.hpp"
#include <algorithm>
#include <dirent.h>
#include <fstream>
#include <iostream>

namespace pipeline {

BinFileReader::BinFileReader(const ReaderConfig& config) : config_(config) {}

bool BinFileReader::open() {
    file_list_.clear();
    current_index_ = 0;

    DIR* dir = opendir(config_.input_path.c_str());
    if (!dir) {
        // Try as single file
        std::ifstream f(config_.input_path, std::ios::binary);
        if (f.is_open()) {
            file_list_.push_back(config_.input_path);
            is_open_ = true;
            return true;
        }
        LOG_ERROR_FMT("Cannot open: {}", config_.input_path);
        return false;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        if (name.size() > 4 && name.substr(name.size() - 4) == ".bin") {
            file_list_.push_back(config_.input_path + "/" + name);
        }
    }
    closedir(dir);

    std::sort(file_list_.begin(), file_list_.end());
    is_open_ = !file_list_.empty();
    return is_open_;
}

bool BinFileReader::read(PointCloud& cloud) {
    if (!is_open_ || current_index_ >= file_list_.size()) {
        return false;
    }

    const std::string& filepath = file_list_[current_index_];

    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return false;

    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t num_points = file_size / (4 * sizeof(float));
    cloud.points.resize(num_points);
    file.read(reinterpret_cast<char*>(cloud.points.data()), file_size);

    // Extract frame_id from filename (e.g., "000000.bin" -> 0)
    size_t slash = filepath.find_last_of('/');
    std::string filename = (slash != std::string::npos) ? filepath.substr(slash + 1) : filepath;
    cloud.frame_id = std::stoull(filename.substr(0, 6));
    cloud.timestamp_ns = 0;

    current_index_++;
    return true;
}

bool BinFileReader::isOpen() const { return is_open_; }

void BinFileReader::close() {
    is_open_ = false;
    current_index_ = 0;
}

}  // namespace pipeline
