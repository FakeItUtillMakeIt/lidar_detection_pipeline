#ifndef __READER_HPP__
#define __READER_HPP__

#include <memory>
#include "types.hpp"

namespace pipeline {

class PointCloudReader {
public:
    virtual ~PointCloudReader() = default;
    virtual bool open() = 0;
    virtual bool read(PointCloud& cloud) = 0;
    virtual bool isOpen() const = 0;
    virtual void close() = 0;
    virtual ReaderType type() const = 0;
};

std::shared_ptr<PointCloudReader> createReader(const ReaderConfig& config);

}  // namespace pipeline

#endif  // __READER_HPP__
