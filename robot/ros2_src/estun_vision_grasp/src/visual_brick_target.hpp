#pragma once

#include <array>
#include <algorithm>
#include <cmath>
#include <istream>
#include <string>

namespace estun_vision_grasp
{
constexpr double kClosingOffsetRad = 0.78539816339744830962;  // tool0 +X towards +Y, 45 deg
constexpr const char* kVisualBrickTargetPath = "/tmp/estun_brick_oriented_target_base.txt";
constexpr const char* kVisualTapeTargetPath = "/tmp/estun_tape_oriented_target_base.txt";
constexpr const char* kVisualPringlesTargetPath = "/tmp/estun_pringles_oriented_target_base.txt";

struct VisualBrickTarget
{
    double stamp = 0.0;
    std::array<double, 3> xyz{};
    double confidence = 0.0;
    std::array<double, 2> short_axis{};
};

inline bool parseOrientedTarget(
    std::istream& input, double now, VisualBrickTarget& target, std::string& reason,
    const std::string& expected_version, const std::string& expected_label)
{
    std::string version, frame, label, extra;
    VisualBrickTarget value;
    if (!(input >> version >> value.stamp >> frame >> label
          >> value.xyz[0] >> value.xyz[1] >> value.xyz[2]
          >> value.confidence >> value.short_axis[0] >> value.short_axis[1]) || (input >> extra))
    {
        reason = "视觉方向目标无效、缺失或格式错误";
        return false;
    }
    if (version != expected_version || frame != "base_link" || label != expected_label)
    {
        reason = "视觉方向目标版本、坐标系或类别不匹配";
        return false;
    }
    for (double v : {now, value.stamp, value.xyz[0], value.xyz[1], value.xyz[2],
                     value.confidence, value.short_axis[0], value.short_axis[1]})
    {
        if (!std::isfinite(v))
        {
            reason = "视觉方向目标包含非有限数值";
            return false;
        }
    }
    if (now - value.stamp < -0.2 || now - value.stamp > 1.0)
    {
        reason = "视觉方向目标过期（允许 1 秒），请检查识别刷新";
        return false;
    }
    const double norm = std::hypot(value.short_axis[0], value.short_axis[1]);
    if (value.confidence < 0.50 || value.confidence > 1.0 || std::abs(norm - 1.0) > 0.02)
    {
        reason = "视觉方向目标置信度或短轴单位向量非法";
        return false;
    }
    for (double& v : value.short_axis) v /= norm;
    target = value;
    return true;
}

inline bool parseVisualBrickTarget(
    std::istream& input, double now, VisualBrickTarget& target, std::string& reason)
{
    return parseOrientedTarget(input, now, target, reason, "brick_v1", "Foam_Brick");
}

inline bool parseVisualTapeTarget(
    std::istream& input, double now, VisualBrickTarget& target, std::string& reason)
{
    return parseOrientedTarget(input, now, target, reason, "tape_v1", "Scotch_Duct_Tape");
}

inline bool parseVisualPringlesTarget(
    std::istream& input, double now, VisualBrickTarget& target, std::string& reason)
{
    return parseOrientedTarget(input, now, target, reason, "pringles_v1", "Pringles");
}

inline double visualToolYaw(const VisualBrickTarget& target)
{
    // With tool0 +Z down, local +Y is clockwise from local +X in base XY.
    // Consequently the required tool0 +X yaw is short_axis_yaw + 45 deg.
    return std::atan2(target.short_axis[1], target.short_axis[0]) + kClosingOffsetRad;
}

inline double closingAxisDifference(const VisualBrickTarget& first, const VisualBrickTarget& second)
{
    const double dot = first.short_axis[0] * second.short_axis[0] +
                       first.short_axis[1] * second.short_axis[1];
    return std::acos(std::clamp(std::abs(dot), 0.0, 1.0));  // axes are equivalent modulo 180 deg
}

inline bool sameVisualBrickTarget(const VisualBrickTarget& planned, const VisualBrickTarget& latest)
{
    double distance_squared = 0.0;
    for (std::size_t i = 0; i < 3; ++i)
        distance_squared += std::pow(planned.xyz[i] - latest.xyz[i], 2);
    return distance_squared <= 0.010 * 0.010 &&
           closingAxisDifference(planned, latest) <= 5.0 * 3.14159265358979323846 / 180.0;
}
}  // namespace estun_vision_grasp
