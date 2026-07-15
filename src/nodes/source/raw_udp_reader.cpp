#include "3rd_party/log_mgr/log_mgr.h"
#include "raw_udp_reader.hpp"
#include <arpa/inet.h>
#include <cstring>
#include <errno.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>

namespace pipeline {

RawUdpReader::RawUdpReader(const ReaderConfig& config) : config_(config) {
    buf_.reserve(200000);
}

RawUdpReader::~RawUdpReader() { close(); }

bool RawUdpReader::open() {
    sockfd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd_ < 0) {
        LOG_ERROR_FMT("[RawUdpReader] socket() failed: {}", strerror(errno));
        return false;
    }

    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(static_cast<uint16_t>(config_.velodyne_port));

    if (bind(sockfd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        LOG_ERROR_FMT("[RawUdpReader] bind() to port {} failed: {}", config_.velodyne_port, strerror(errno));
        ::close(sockfd_);
        sockfd_ = -1;
        return false;
    }

    // Try to increase receive buffer (kernel may cap to rmem_max)
    int rcvbuf = 4 * 1024 * 1024;
    setsockopt(sockfd_, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));

    int flags = fcntl(sockfd_, F_GETFL, 0);
    fcntl(sockfd_, F_SETFL, flags | O_NONBLOCK);

    LOG_INFO_FMT("[RawUdpReader] Listening on UDP port {}", config_.velodyne_port);
    return true;
}

bool RawUdpReader::isOpen() const { return sockfd_ >= 0; }

void RawUdpReader::close() {
    if (sockfd_ >= 0) {
        ::close(sockfd_);
        sockfd_ = -1;
    }
    buf_.clear();
}

bool RawUdpReader::read(PointCloud& cloud) {
    if (!isOpen()) return false;

    cloud.points.clear();
    cloud.frame_id = 0;

    uint8_t pkt_buf[65535];
    int idle = 0;

    while (idle < 500) {
        // Drain all available packets
        while (true) {
            ssize_t n = recvfrom(sockfd_, pkt_buf, sizeof(pkt_buf), 0, nullptr, nullptr);
            if (n < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) break;
                LOG_ERROR_FMT("[RawUdpReader] recvfrom() error: {}", strerror(errno));
                return false;
            }

            if (static_cast<size_t>(n) < 16) continue;

            // Custom frame protocol: [4B magic][4B frame_id][4B total][4B num]
            uint32_t vals[4];
            if (n < 16) continue;
            std::memcpy(vals, pkt_buf, 16);
            uint32_t magic     = vals[0];
            uint32_t frame_id  = vals[1];
            uint32_t total_pts = vals[2];
            uint32_t num_pts   = vals[3];

            if (magic != 0x41444350) continue;  // 'PCDA'

            if (frame_id != cur_frame_id_) {
                buf_.clear();
                cur_frame_id_ = frame_id;
                expected_total_ = total_pts;
            }

            size_t data_bytes = static_cast<size_t>(n) - 16;
            size_t pt_count = data_bytes / sizeof(PointXYZI);
            const PointXYZI* src = reinterpret_cast<const PointXYZI*>(pkt_buf + 16);
            buf_.insert(buf_.end(), src, src + pt_count);

            if (buf_.size() >= expected_total_) {
                cloud.points.swap(buf_);
                LOG_INFO_FMT("[RawUdpReader] Frame {}: {} points",
                             cur_frame_id_, cloud.points.size());
                cur_frame_id_ = 0xFFFFFFFF;
                expected_total_ = 0;
                return true;
            }
        }

        if (buf_.empty()) return false;

        usleep(1000);  // 1ms 快速轮询下一批
        idle++;
    }

    LOG_WARN_FMT("[RawUdpReader] Timeout: got {}/{} pts, frame {}",
                 buf_.size(), expected_total_, cur_frame_id_);
    buf_.clear();
    cur_frame_id_ = 0xFFFFFFFF;
    expected_total_ = 0;
    return false;
}

}  // namespace pipeline
