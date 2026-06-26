// src/nodes/track/tracker_node.h
#pragma once

#include "lidar_core/nodes/i_tracker_node.h"
#include "ocsort_adapter.h"
#include <memory>

namespace lidar_core {
namespace nodes {

class TrackerNode : public ITrackerNode {
public:
    TrackerNode();
    ~TrackerNode() override;

    // ITrackerNode 接口
    void setTrackerType(TrackerType type) override;
    TrackerType getTrackerType() const override;
    void setOCSortConfig(const OCSortConfig& config) override;
    int getActiveTrackCount() const override;

    // Node 接口
    bool start() override;
    void stop() override;
    void pushData(std::shared_ptr<core::BasePacket> packet) override;

private:
    TrackerType tracker_type_ = TrackerType::OCSORT;
    OCSortConfig ocsort_config_;
    std::unique_ptr<OCSortAdapter> adapter_;
};

} // namespace nodes
} // namespace lidar_core
