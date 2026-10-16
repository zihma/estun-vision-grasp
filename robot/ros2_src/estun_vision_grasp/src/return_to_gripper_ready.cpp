#include "gripper_ready_target.hpp"

#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <rclcpp/rclcpp.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using MoveGroup = moveit::planning_interface::MoveGroupInterface;
using namespace std::chrono_literals;

namespace
{
constexpr std::size_t N = estun_vision_grasp::kArmJointCount;

// 真机验证过的 blended 方法：每个逻辑分段最多变化 0.10 rad。
constexpr double STEP_MAX = 0.10;
constexpr std::size_t MAX_SEGMENTS = 20;
constexpr double READY_TOLERANCE = 0.01;
constexpr double FINAL_ACTUAL_TOLERANCE = 0.03;
constexpr double START_STATE_TOLERANCE = 0.03;
constexpr double MAX_PLAN_VELOCITY = 0.095;
constexpr double MAX_PLAN_ACCELERATION = 0.095;
constexpr double PLAN_FINAL_TOLERANCE = 0.005;

const std::array<std::string, N> JOINT_NAMES = {
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"};
using JointArray = std::array<double, N>;

double maxDifference(const JointArray &first, const JointArray &second)
{
    double result = 0.0;
    for (std::size_t i = 0; i < N; ++i)
        result = std::max(result, std::abs(first[i] - second[i]));
    return result;
}

bool readCurrentJoints(
    MoveGroup &group,
    JointArray &joints,
    moveit::core::RobotStatePtr &state,
    const rclcpp::Logger &logger)
{
    state = group.getCurrentState(5.0);
    if (!state)
    {
        RCLCPP_ERROR(logger, "5 秒内没有取得当前机械臂状态");
        return false;
    }
    for (std::size_t i = 0; i < N; ++i)
    {
        joints[i] = state->getVariablePosition(JOINT_NAMES[i]);
        if (!std::isfinite(joints[i]))
        {
            RCLCPP_ERROR(logger, "%s 当前值不是有限数", JOINT_NAMES[i].c_str());
            return false;
        }
    }
    return true;
}

JointArray nextWaypoint(const JointArray &current)
{
    const double error =
        maxDifference(current, estun_vision_grasp::kGripperReady);
    if (error <= STEP_MAX)
        return estun_vision_grasp::kGripperReady;

    const double ratio = STEP_MAX / error;
    JointArray target{};
    for (std::size_t i = 0; i < N; ++i)
    {
        target[i] = current[i] + ratio *
            (estun_vision_grasp::kGripperReady[i] - current[i]);
    }
    return target;
}

bool planSmallSegment(
    MoveGroup &group,
    const moveit::core::RobotStatePtr &state,
    const JointArray &start,
    const JointArray &target,
    MoveGroup::Plan &plan,
    const rclcpp::Logger &logger)
{
    for (std::size_t i = 0; i < N; ++i)
        state->setVariablePosition(JOINT_NAMES[i], start[i]);
    state->update();
    group.setStartState(*state);

    const std::vector<double> target_vector(target.begin(), target.end());
    if (!group.setJointValueTarget(target_vector))
    {
        RCLCPP_ERROR(logger, "setJointValueTarget 失败");
        return false;
    }

    plan = MoveGroup::Plan();
    if (group.plan(plan) != moveit::core::MoveItErrorCode::SUCCESS)
    {
        RCLCPP_ERROR(logger, "本段规划失败");
        return false;
    }

    const auto &trajectory = plan.trajectory.joint_trajectory;
    if (trajectory.points.empty() || trajectory.joint_names.size() != N)
    {
        RCLCPP_ERROR(logger, "本段规划轨迹为空或关节数异常");
        return false;
    }

    std::unordered_map<std::string, std::size_t> index;
    for (std::size_t i = 0; i < trajectory.joint_names.size(); ++i)
        index[trajectory.joint_names[i]] = i;
    for (const auto &name : JOINT_NAMES)
    {
        if (!index.count(name))
        {
            RCLCPP_ERROR(logger, "本段规划轨迹缺少 %s", name.c_str());
            return false;
        }
    }

    double max_velocity = 0.0;
    double max_acceleration = 0.0;
    for (const auto &point : trajectory.points)
    {
        if (point.positions.size() != N)
        {
            RCLCPP_ERROR(logger, "本段轨迹点关节数异常");
            return false;
        }
        for (std::size_t i = 0; i < N; ++i)
        {
            const std::size_t j = index.at(JOINT_NAMES[i]);
            if (!std::isfinite(point.positions[j]))
            {
                RCLCPP_ERROR(logger, "本段轨迹包含非有限关节值");
                return false;
            }
            if (!point.velocities.empty())
                max_velocity = std::max(max_velocity, std::abs(point.velocities[j]));
            if (!point.accelerations.empty())
                max_acceleration = std::max(max_acceleration, std::abs(point.accelerations[j]));
        }
    }

    JointArray planned_final{};
    const auto &last = trajectory.points.back();
    for (std::size_t i = 0; i < N; ++i)
        planned_final[i] = last.positions[index.at(JOINT_NAMES[i])];

    const double requested = maxDifference(start, target);
    const double final_error = maxDifference(planned_final, target);
    RCLCPP_INFO(
        logger,
        "segment: request=%.6f MoveIt_points=%zu vel=%.6f acc=%.6f final_error=%.6f",
        requested, trajectory.points.size(), max_velocity, max_acceleration, final_error);

    if (requested > STEP_MAX + 1.0e-9)
    {
        RCLCPP_ERROR(logger, "逻辑分段超过 %.3f rad", STEP_MAX);
        return false;
    }
    if (max_velocity > MAX_PLAN_VELOCITY + 1.0e-9)
    {
        RCLCPP_ERROR(logger, "本段规划速度超限");
        return false;
    }
    if (max_acceleration > MAX_PLAN_ACCELERATION + 1.0e-9)
    {
        RCLCPP_ERROR(logger, "本段规划加速度超限");
        return false;
    }
    if (final_error > PLAN_FINAL_TOLERANCE)
    {
        RCLCPP_ERROR(logger, "本段规划终点误差超限");
        return false;
    }
    return true;
}
}  // namespace

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    const auto node = rclcpp::Node::make_shared("return_to_gripper_ready");
    const auto logger = node->get_logger();

    const bool execute = node->declare_parameter<bool>("execute", false);
    const bool require_confirmation =
        node->declare_parameter<bool>("require_confirmation", true);
    const double max_joint_delta =
        node->declare_parameter<double>("max_joint_delta", 1.50);
    const double velocity_scaling =
        node->declare_parameter<double>("velocity_scaling", 0.03);
    const double acceleration_scaling =
        node->declare_parameter<double>("acceleration_scaling", 0.03);
    const double planning_time =
        node->declare_parameter<double>("planning_time", 5.0);
    const int planning_attempts =
        node->declare_parameter<int>("planning_attempts", 10);

    if (!std::isfinite(max_joint_delta) || max_joint_delta < 0.10 ||
        max_joint_delta > STEP_MAX * static_cast<double>(MAX_SEGMENTS))
    {
        RCLCPP_ERROR(logger, "max_joint_delta 必须在 0.10～2.00 rad");
        rclcpp::shutdown();
        return 1;
    }
    if (!std::isfinite(velocity_scaling) || velocity_scaling <= 0.0 ||
        velocity_scaling > 0.095 || !std::isfinite(acceleration_scaling) ||
        acceleration_scaling <= 0.0 || acceleration_scaling > 0.095)
    {
        RCLCPP_ERROR(logger, "速度和加速度缩放必须在 (0, 0.095] 范围内");
        rclcpp::shutdown();
        return 1;
    }
    if (!std::isfinite(planning_time) || planning_time < 1.0 ||
        planning_time > 60.0 || planning_attempts < 1 || planning_attempts > 100)
    {
        RCLCPP_ERROR(logger, "规划时间或规划次数参数非法");
        rclcpp::shutdown();
        return 1;
    }

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    std::thread spin_thread([&executor]() { executor.spin(); });
    auto finish = [&](int code) {
        executor.cancel();
        if (spin_thread.joinable()) spin_thread.join();
        rclcpp::shutdown();
        return code;
    };

    MoveGroup group(node, "arm");
    group.setPlanningTime(planning_time);
    group.setNumPlanningAttempts(planning_attempts);
    group.setMaxVelocityScalingFactor(velocity_scaling);
    group.setMaxAccelerationScalingFactor(acceleration_scaling);

    JointArray current{};
    moveit::core::RobotStatePtr state;
    if (!readCurrentJoints(group, current, state, logger)) return finish(1);

    RCLCPP_INFO(logger, "========== BLENDED GRIPPER READY ==========");
    for (std::size_t i = 0; i < N; ++i)
    {
        RCLCPP_INFO(
            logger, "%s current=% .9f goal=% .9f delta=% .9f",
            JOINT_NAMES[i].c_str(), current[i],
            estun_vision_grasp::kGripperReady[i],
            estun_vision_grasp::kGripperReady[i] - current[i]);
    }

    const double initial_delta =
        maxDifference(current, estun_vision_grasp::kGripperReady);
    RCLCPP_INFO(logger, "最大单关节变化 = %.9f rad", initial_delta);
    if (initial_delta <= READY_TOLERANCE)
    {
        RCLCPP_INFO(logger, "当前已经位于 Gripper Ready");
        return finish(0);
    }
    if (execute && initial_delta > max_joint_delta)
    {
        RCLCPP_ERROR(
            logger, "目标关节变化过大：%.6f > %.6f rad，拒绝执行",
            initial_delta, max_joint_delta);
        return finish(2);
    }

    std::vector<JointArray> waypoints;
    JointArray theoretical = current;
    MoveGroup::Plan first_plan;
    bool have_first_plan = false;
    for (std::size_t segment = 0; segment < MAX_SEGMENTS; ++segment)
    {
        if (maxDifference(theoretical, estun_vision_grasp::kGripperReady) <=
            READY_TOLERANCE) break;

        const JointArray target = nextWaypoint(theoretical);
        MoveGroup::Plan segment_plan;
        if (!planSmallSegment(
                group, state, theoretical, target, segment_plan, logger))
        {
            RCLCPP_ERROR(logger, "Segment %zu 检查失败，不执行", segment + 1);
            return finish(1);
        }
        if (!have_first_plan)
        {
            first_plan = segment_plan;
            have_first_plan = true;
        }
        waypoints.push_back(target);
        RCLCPP_INFO(
            logger,
            "W%02zu delta=%.9f q=[%.6f %.6f %.6f %.6f %.6f %.6f]",
            segment + 1, maxDifference(theoretical, target),
            target[0], target[1], target[2], target[3], target[4], target[5]);
        theoretical = target;
    }

    if (!have_first_plan || waypoints.empty())
    {
        RCLCPP_ERROR(logger, "没有生成有效 waypoint");
        return finish(1);
    }
    if (maxDifference(theoretical, estun_vision_grasp::kGripperReady) >
        READY_TOLERANCE)
    {
        RCLCPP_ERROR(logger, "MAX_SEGMENTS=%zu 不足以到达 Gripper Ready", MAX_SEGMENTS);
        return finish(1);
    }

    MoveGroup::Plan combined;
    combined.start_state = first_plan.start_state;
    auto &joint_trajectory = combined.trajectory.joint_trajectory;
    joint_trajectory.joint_names =
        first_plan.trajectory.joint_trajectory.joint_names;

    std::unordered_map<std::string, std::size_t> canonical;
    for (std::size_t i = 0; i < N; ++i) canonical[JOINT_NAMES[i]] = i;
    auto makePoint = [&](const JointArray &joints, int seconds) {
        trajectory_msgs::msg::JointTrajectoryPoint point;
        for (const auto &name : joint_trajectory.joint_names)
        {
            if (!canonical.count(name))
                throw std::runtime_error("轨迹中出现未知关节名");
            point.positions.push_back(joints[canonical.at(name)]);
        }
        // real_robot_bridge 会按已验证协议写入 REAL_VEL / REAL_ACC。
        point.velocities.assign(N, 0.0);
        point.accelerations.assign(N, 0.0);
        point.time_from_start.sec = seconds;
        point.time_from_start.nanosec = 0;
        return point;
    };

    try
    {
        joint_trajectory.points.push_back(makePoint(current, 0));
        for (std::size_t i = 0; i < waypoints.size(); ++i)
            joint_trajectory.points.push_back(
                makePoint(waypoints[i], static_cast<int>(i + 1)));
    }
    catch (const std::exception &error)
    {
        RCLCPP_ERROR(logger, "构造 combined 轨迹失败：%s", error.what());
        return finish(1);
    }

    RCLCPP_INFO(logger, "========== COMBINED SUMMARY ==========");
    RCLCPP_INFO(logger, "逻辑运动段数 = %zu", waypoints.size());
    RCLCPP_INFO(logger, "发送 trajectory points = %zu", joint_trajectory.points.size());
    RCLCPP_INFO(logger, "最大逻辑分段 = %.3f rad", STEP_MAX);
    RCLCPP_INFO(
        logger, "最终目标误差 = %.9f rad",
        maxDifference(waypoints.back(), estun_vision_grasp::kGripperReady));

    if (!execute)
    {
        RCLCPP_WARN(logger, "PLAN-ONLY 完成：没有向真机发送运动指令");
        return finish(0);
    }

    if (require_confirmation)
    {
        std::cout
            << "\n========== BLENDED GRIPPER READY 确认 ==========\n"
            << "确认 ESTUN 当前为 AutoReady，fault_flag=0\n"
            << "确认整条路径无人、无障碍物，急停可立即使用\n"
            << "确认以上所有 0.1 rad segment 检查均正常\n\n"
            << "准确输入 GRIPPER_READY 才执行完整连续回位："
            << std::flush;
        std::string answer;
        std::cin >> answer;
        if (answer != "GRIPPER_READY")
        {
            RCLCPP_WARN(logger, "口令不匹配，取消执行");
            return finish(0);
        }
    }

    JointArray before_execute{};
    moveit::core::RobotStatePtr before_execute_state;
    if (!readCurrentJoints(
            group, before_execute, before_execute_state, logger)) return finish(1);
    const double start_change = maxDifference(current, before_execute);
    if (start_change > START_STATE_TOLERANCE)
    {
        RCLCPP_ERROR(
            logger, "确认期间机械臂发生移动：%.6f > %.6f rad，旧规划作废",
            start_change, START_STATE_TOLERANCE);
        return finish(2);
    }

    RCLCPP_WARN(
        logger, "开始 blended Execute：%zu 个逻辑段，%zu 个轨迹点",
        waypoints.size(), joint_trajectory.points.size());
    const auto execute_result = group.execute(combined);
    if (execute_result != moveit::core::MoveItErrorCode::SUCCESS)
    {
        RCLCPP_ERROR(logger, "Gripper Ready blended Execute 失败或被中止");
        return finish(1);
    }

    std::this_thread::sleep_for(250ms);
    JointArray actual{};
    moveit::core::RobotStatePtr final_state;
    if (!readCurrentJoints(group, actual, final_state, logger))
    {
        RCLCPP_ERROR(logger, "执行后无法读取实际状态");
        return finish(1);
    }

    const double actual_error =
        maxDifference(actual, estun_vision_grasp::kGripperReady);
    RCLCPP_INFO(logger, "========== FINAL RESULT ==========");
    for (std::size_t i = 0; i < N; ++i)
    {
        RCLCPP_INFO(
            logger, "%s actual=% .9f goal=% .9f error=% .9f",
            JOINT_NAMES[i].c_str(), actual[i],
            estun_vision_grasp::kGripperReady[i],
            actual[i] - estun_vision_grasp::kGripperReady[i]);
    }
    RCLCPP_INFO(logger, "Gripper Ready 最大实际误差 = %.9f rad", actual_error);
    if (actual_error > FINAL_ACTUAL_TOLERANCE)
    {
        RCLCPP_ERROR(logger, "最终实际误差超过 %.3f rad", FINAL_ACTUAL_TOLERANCE);
        return finish(1);
    }

    RCLCPP_INFO(logger, "BLENDED GRIPPER READY 完成");
    return finish(0);
}
