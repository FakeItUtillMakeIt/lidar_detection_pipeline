#ifndef __VELODYNE_READER_HPP__
#define __VELODYNE_READER_HPP__

#include "reader.hpp"

namespace pipeline {

class VelodyneReader : public PointCloudReader {
public:
    explicit VelodyneReader(const ReaderConfig& config);
    ~VelodyneReader() override;

    bool open() override;
    bool read(PointCloud& cloud) override;
    bool isOpen() const override;
    void close() override;
    ReaderType type() const override { return ReaderType::VELODYNE_UDP; }

private:
    ReaderConfig config_;
    int sock_fd_ = -1;
    bool is_open_ = false;
};

}  // namespace pipeline

#endif  // __VELODYNE_READER_HPP__
