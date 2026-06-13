#include "velodyne_reader.hpp"
#include <iostream>

namespace pipeline {

VelodyneReader::VelodyneReader(const ReaderConfig& config) : config_(config) {}

VelodyneReader::~VelodyneReader() { close(); }

bool VelodyneReader::open() {
    // TODO: Create UDP socket and bind to config_.velodyne_port
    std::cout << "[VelodyneReader] UDP socket not yet implemented" << std::endl;
    std::cout << "[VelodyneReader] Port: " << config_.velodyne_port << std::endl;
    return false;
}

bool VelodyneReader::read(PointCloud& cloud) {
    // TODO: Receive UDP packets, decode Velodyne data, fill cloud
    return false;
}

bool VelodyneReader::isOpen() const { return is_open_; }

void VelodyneReader::close() {
    if (sock_fd_ >= 0) {
        // close(sock_fd_);
        sock_fd_ = -1;
    }
    is_open_ = false;
}

}  // namespace pipeline
