// include/lidar_core/nodes/i_planner_node.h
#pragma once

#include "lidar_core/core/node.h"
#include "lidar_core/core/planning_types.hpp"
#include <string>

namespace lidar_core {
namespace nodes {

// 规划器类型枚举
enum class PlannerType {
    POTENTIAL_FIELD = 0,    // 人工势场法
    LATTICE = 1,            // Lattice规划
    OPTIMIZATION = 2,       // 优化规划
};

// 规划节点接口
class IPlannerNode : public core::Node {
public:
    using core::Node::Node;

    // 设置规划器类型
    virtual void setPlannerType(PlannerType type) = 0;
    virtual PlannerType getPlannerType() const = 0;

    // 设置规划配置
    virtual void setPlannerConfig(const core::PlanningConfig& config) = 0;
    virtual const core::PlanningConfig& getPlannerConfig() const = 0;

    // 获取最新规划结果
    virtual const core::Trajectory& getLatestTrajectory() const = 0;

    // 获取规划状态
    virtual bool isPlanningFeasible() const = 0;
};

} // namespace nodes
} // namespace lidar_core