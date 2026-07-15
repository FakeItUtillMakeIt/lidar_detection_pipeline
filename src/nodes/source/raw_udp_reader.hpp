#ifndef __RAW_UDP_READER_HPP__
#define __RAW_UDP_READER_HPP__

#include "reader.hpp"
#include <cstdint>
#include <vector>

namespace pipeline {

class RawUdpReader : public PointCloudReader {
public:
    explicit RawUdpReader(const ReaderConfig& config);
    ~RawUdpReader() override;

    bool open() override;
    bool read(PointCloud& cloud) override;
    bool isOpen() const override;
    void close() override;
    ReaderType type() const override { return ReaderType::RAW_UDP; }

private:
    ReaderConfig config_;
    int sockfd_ = -1;

    std::vector<PointXYZI> buf_;
    uint32_t cur_frame_id_ = 0xFFFFFFFF;
    uint32_t expected_total_ = 0;
};

}  // namespace pipeline

#endif  // __RAW_UDP_READER_HPP__
