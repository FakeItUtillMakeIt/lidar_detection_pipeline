#ifndef __VELODYNE_READER_HPP__
#define __VELODYNE_READER_HPP__

#include "reader.hpp"
#include <cstdint>

namespace pipeline {

class VelodyneUdpReader : public PointCloudReader {
public:
    explicit VelodyneUdpReader(const ReaderConfig& config);
    ~VelodyneUdpReader() override;

    bool open() override;
    bool read(PointCloud& cloud) override;
    bool isOpen() const override;
    void close() override;
    ReaderType type() const override { return ReaderType::VELODYNE_UDP; }

private:
    bool parsePacket(const uint8_t* data, size_t len, PointCloud& cloud);
    void addPoint(float azimuth, uint8_t laser_id, uint16_t dist, uint8_t intensity,
                  uint32_t timestamp, PointCloud& cloud);

    ReaderConfig config_;
    int sockfd_ = -1;

    // VLP-16 laser vertical angles (degrees): interleaved upper/lower bank
    static constexpr float kLaserAngles[16] = {
        -15.0f, 1.0f, -13.0f, 3.0f, -11.0f, 5.0f, -9.0f, 7.0f,
        -7.0f,  9.0f, -5.0f, 11.0f, -3.0f, 13.0f, -1.0f, 15.0f
    };
    static constexpr int kMaxPointsPerRead = 100000;
};

}  // namespace pipeline

#endif  // __VELODYNE_READER_HPP__
