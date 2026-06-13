#include "file_writer.hpp"
#include <sys/stat.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace pipeline {

bool FileWriter::init(const OutputConfig& config) {
    output_dir_ = config.output_dir;
    mkdir(output_dir_.c_str(), 0755);
    return true;
}

void FileWriter::write(const std::vector<Detection>& dets, uint64_t frame_id) {
    std::ostringstream oss;
    oss << output_dir_ << "/" << std::setfill('0') << std::setw(6) << frame_id << ".txt";

    std::ofstream ofs(oss.str());
    if (!ofs.is_open()) {
        std::cerr << "Cannot open output file: " << oss.str() << std::endl;
        return;
    }

    for (const auto& det : dets) {
        ofs << det.x << " " << det.y << " " << det.z << " "
            << det.w << " " << det.l << " " << det.h << " "
            << det.rt << " " << det.class_id << " " << det.score << "\n";
    }
}

void FileWriter::close() {}

}  // namespace pipeline
