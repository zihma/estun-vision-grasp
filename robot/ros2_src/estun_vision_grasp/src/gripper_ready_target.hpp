#pragma once

#include <array>
#include <cstddef>

namespace estun_vision_grasp
{

inline constexpr std::size_t kArmJointCount = 6;

// 已在真机示教确认的四指夹爪朝下姿态，单位为 MoveIt 弧度。
inline constexpr std::array<double, kArmJointCount> kGripperReady = {
    0.5651725184,
    0.4872435673,
    1.6332267541,
   -0.2517462913,
   -1.8531382399,
    0.1293987107
};

}  // namespace estun_vision_grasp
