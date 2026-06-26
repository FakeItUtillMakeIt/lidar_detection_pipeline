#ifndef __BIN_READER_HPP__
#define __BIN_READER_HPP__

#include "reader.hpp"
#include <vector>
#include <string>

namespace pipeline {

class BinFileReader : public PointCloudReader {
public:
    explicit BinFileReader(const ReaderConfig& config);
    ~BinFileReader() override = default;

    bool open() override;
    bool read(PointCloud& cloud) override;
    bool isOpen() const override;
    void close() override;
    ReaderType type() const override { return ReaderType::BIN_FILE; }

private:
    ReaderConfig config_;
    std::vector<std::string> file_list_;
    size_t current_index_ = 0;
    bool is_open_ = false;
};

}  // namespace pipeline

#endif  // __BIN_READER_HPP__
