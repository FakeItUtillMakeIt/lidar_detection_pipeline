#include "3rd_party/log_mgr/log_mgr.h"
#include "velodyne_reader.hpp"
#include <arpa/inet.h>
#include <cmath>
#include <cstring>
#include <errno.h>
#include <sys/socket.h>
#include <unistd.h>

namespace pipeline {

// VLP-16 constants
static constexpr int kPacketHeaderLen = 42;
static constexpr int kBlockSize = 100;          // bytes per data block
static constexpr int kBlocksPerPacket = 12;
static constexpr int kLasersPerBlock = 16;      // 16 lasers in single return
static constexpr int kMeasurementSize = 3;      // 2 bytes distance + 1 byte intensity
static constexpr uint16_t kBlockFlag = 0xEEFF;  // VLP-16 block identifier
static constexpr float kDistScale = 0.002f;     // distance unit = 2mm
static constexpr float kRadPerUnit = 3.14159265f / 18000.0f;  // 0.01° to rad
static constexpr int kMaxRecvSize = 2000;

VelodyneUdpReader::VelodyneUdpReader(const ReaderConfig& config) : config_(config) {}

VelodyneUdpReader::~VelodyneUdpReader() { close(); }

bool VelodyneUdpReader::open() {
    sockfd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd_ < 0) {
        LOG_ERROR_FMT("[VelodyneReader] socket() failed: {}", strerror(errno));
        return false;
    }

    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(static_cast<uint16_t>(config_.velodyne_port));

    if (bind(sockfd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        LOG_ERROR_FMT("[VelodyneReader] bind() to port {} failed: {}", config_.velodyne_port, strerror(errno));
        ::close(sockfd_);
        sockfd_ = -1;
        return false;
    }

    // 100ms receive timeout so read() returns periodically with accumulated points
    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 100000;
    setsockopt(sockfd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    LOG_INFO_FMT("[VelodyneReader] Listening on UDP port {}", config_.velodyne_port);
    return true;
}

bool VelodyneUdpReader::isOpen() const { return sockfd_ >= 0; }

void VelodyneUdpReader::close() {
    if (sockfd_ >= 0) {
        ::close(sockfd_);
        sockfd_ = -1;
    }
}

bool VelodyneUdpReader::read(PointCloud& cloud) {
    if (!isOpen()) return false;

    cloud.points.clear();
    cloud.points.reserve(kMaxPointsPerRead);
    cloud.frame_id = 0;

    uint8_t buf[kMaxRecvSize];
    int total_packets = 0;

    // Receive packets until timeout or max points reached
    while (static_cast<int>(cloud.points.size()) < kMaxPointsPerRead) {
        struct sockaddr_in sender;
        socklen_t sender_len = sizeof(sender);
        ssize_t n = recvfrom(sockfd_, buf, sizeof(buf), 0,
                             (struct sockaddr*)&sender, &sender_len);
        if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                break;  // timeout → return accumulated points
            }
            LOG_ERROR_FMT("[VelodyneReader] recvfrom() error: {}", strerror(errno));
            return !cloud.points.empty();
        }
        if (static_cast<size_t>(n) < kPacketHeaderLen + 4) continue;

        total_packets++;
        if (!parsePacket(buf, static_cast<size_t>(n), cloud)) {
            LOG_WARN_FMT("[VelodyneReader] Invalid packet ({} bytes)", n);
        }
    }

    LOG_INFO_FMT("[VelodyneReader] Received {} packets, {} points", total_packets, cloud.points.size());
    return !cloud.points.empty();
}

bool VelodyneUdpReader::parsePacket(const uint8_t* data, size_t len, PointCloud& cloud) {
    // Extract timestamp (μs since top of hour, at fixed offset from end)
    uint32_t timestamp = 0;
    if (len >= 6) {
        timestamp = (static_cast<uint32_t>(data[len - 6]) << 24) |
                    (static_cast<uint32_t>(data[len - 5]) << 16) |
                    (static_cast<uint32_t>(data[len - 4]) << 8) |
                    static_cast<uint32_t>(data[len - 3]);
    }

    // Parse 12 data blocks starting after header
    for (int bi = 0; bi < kBlocksPerPacket; bi++) {
        size_t offset = kPacketHeaderLen + bi * kBlockSize;
        if (offset + 4 > len) break;

        uint16_t flag = (static_cast<uint16_t>(data[offset + 1]) << 8) |
                         static_cast<uint16_t>(data[offset]);
        if (flag != kBlockFlag) continue;

        uint16_t azimuth_raw = (static_cast<uint16_t>(data[offset + 3]) << 8) |
                                static_cast<uint16_t>(data[offset + 2]);
        float azimuth = azimuth_raw * kRadPerUnit;

        // Parse 16 laser measurements
        for (int li = 0; li < kLasersPerBlock; li++) {
            size_t moffset = offset + 4 + li * kMeasurementSize;
            if (moffset + kMeasurementSize > len) break;

            uint16_t dist = (static_cast<uint16_t>(data[moffset + 1]) << 8) |
                             static_cast<uint16_t>(data[moffset]);
            uint8_t intensity = data[moffset + 2];

            if (dist == 0) continue;  // no return
            addPoint(azimuth, li, dist, intensity, timestamp, cloud);
        }
    }
    return true;
}

void VelodyneUdpReader::addPoint(float azimuth, uint8_t laser_id, uint16_t dist_units,
                                   uint8_t intensity, uint32_t timestamp,
                                   PointCloud& cloud) {
    if (laser_id >= 16) return;

    float distance = dist_units * kDistScale;
    float vert_angle = kLaserAngles[laser_id] * (3.14159265f / 180.0f);

    float cos_vert = std::cos(vert_angle);
    float sin_vert = std::sin(vert_angle);
    float cos_azim = std::cos(azimuth);
    float sin_azim = std::sin(azimuth);

    PointXYZI pt;
    pt.x = distance * cos_vert * cos_azim;
    pt.y = -distance * cos_vert * sin_azim;  // LiDAR y is left, sensor y is right
    pt.z = distance * sin_vert;
    pt.intensity = intensity / 255.0f;

    cloud.points.push_back(pt);
    cloud.timestamp_ns = static_cast<uint64_t>(timestamp) * 1000;
}

}  // namespace pipeline
