#include "gripper_ready_target.hpp"
#include "visual_brick_target.hpp"

#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit/robot_state/robot_state.hpp>

#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <moveit_msgs/msg/move_it_error_codes.hpp>

#include <rclcpp/rclcpp.hpp>
#include <codroid_msgs/msg/robot_info.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <step_motor/msg/motor.hpp>

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <mutex>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

using MoveGroup =
    moveit::planning_interface::MoveGroupInterface;

using namespace std::chrono_literals;

namespace
{

constexpr std::size_t N = 6;

// ------------------------------------------------------------
// 大范围关节运动：沿用 blended Ready 成功参数
// ------------------------------------------------------------

constexpr double JOINT_STEP_MAX = 0.10;
// 已验证的 blended 方案：每段独立规划检查，最后合并为
// 一次连续 RobotMove。与 return_to_gripper_ready 保持一致：
// 总段数保护上限为 20；连同起点最多发送 21 个点。
constexpr std::size_t MAX_SEGMENTS = 20;

constexpr double MAX_VEL = 0.095;
constexpr double MAX_ACC = 0.095;

constexpr double PLAN_FINAL_TOL = 0.005;
constexpr double JOINT_GOAL_TOL = 0.01;
constexpr double FINAL_ACTUAL_TOL = 0.03;
constexpr double START_STATE_TOL = 0.03;


// ------------------------------------------------------------
// Cartesian：沿用已成功的 real_cartesian_down_up
// ------------------------------------------------------------

constexpr double EEF_STEP = 0.005;
constexpr double CART_SEGMENT = 0.020;

constexpr double CART_MAX_DELTA = 0.30;
constexpr double CART_MAX_VEL = 0.10;
constexpr double CART_MAX_ACC = 0.10;

// 真机桥接层默认使用 0.10。仅垂直下降/抬升在完成 MoveIt
// 原始轨迹安全检查后显式请求 0.20；大范围关节运动仍保持 0.10。
constexpr double CART_REAL_VEL = 0.20;
constexpr double CART_REAL_ACC = 0.20;

constexpr double FRACTION_REQUIRED = 0.999999;

// 完整抓取：先到砖块中心正上方 150 mm，再沿 base_link -Z
// 下降 150 mm；夹紧后沿原路上升 150 mm。
constexpr double APPROACH_DISTANCE = 0.150;
constexpr double LIFT_DISTANCE = APPROACH_DISTANCE;


// ------------------------------------------------------------

const std::array<std::string, N> NAMES = {
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6"
};

// ============================================================
// 通用
// ============================================================

double maxDiff(
    const std::array<double, N>& a,
    const std::array<double, N>& b)
{
    double d = 0.0;

    for (std::size_t i = 0; i < N; ++i)
    {
        d = std::max(
            d,
            std::abs(a[i] - b[i])
        );
    }

    return d;
}


bool readCurrent(
    MoveGroup& group,
    std::array<double, N>& q,
    moveit::core::RobotStatePtr& state)
{
    state = group.getCurrentState(5.0);

    if (!state)
        return false;

    for (std::size_t i = 0; i < N; ++i)
    {
        q[i] =
            state->getVariablePosition(
                NAMES[i]
            );

        if (!std::isfinite(q[i]))
            return false;
    }

    return true;
}


// ============================================================
// 视觉目标
// ============================================================

bool readTarget(
    double& x,
    double& y,
    double& z,
    double& confidence)
{
    std::ifstream fin(
        "/tmp/estun_foam_target_base.txt"
    );

    if (!fin.is_open())
    {
        std::cerr
            << "找不到 /tmp/estun_foam_target_base.txt\n";

        return false;
    }

    double stamp = 0.0;
    std::string frame;

    fin
        >> stamp
        >> frame
        >> x
        >> y
        >> z
        >> confidence;

    if (!fin)
    {
        std::cerr
            << "estun_foam_target_base.txt 格式错误\n";

        return false;
    }

    const double now =
        std::chrono::duration<double>(
            std::chrono::system_clock::now()
                .time_since_epoch()
        ).count();

    const double age =
        now - stamp;

    if (age < -0.2 || age > 1.0)
    {
        std::cerr
            << "视觉目标过期："
            << age
            << " s\n";

        return false;
    }

    if (frame != "base_link")
    {
        std::cerr
            << "拒绝非 MoveIt base_link 目标："
            << frame << "\n";

        return false;
    }

    if (confidence < 0.50)
    {
        std::cerr
            << "目标置信度过低："
            << confidence << "\n";

        return false;
    }

    return
        std::isfinite(x) &&
        std::isfinite(y) &&
        std::isfinite(z);
}

enum class OrientedTargetKind { Brick, Tape, Pringles };

bool readVisualTarget(estun_vision_grasp::VisualBrickTarget& target,
                      const rclcpp::Logger& logger, OrientedTargetKind kind)
{
    const char* path = kind == OrientedTargetKind::Tape ? estun_vision_grasp::kVisualTapeTargetPath :
                       kind == OrientedTargetKind::Pringles ? estun_vision_grasp::kVisualPringlesTargetPath :
                                                                estun_vision_grasp::kVisualBrickTargetPath;
    std::ifstream input(path);
    const double now = std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    std::string reason;
    const bool parsed = kind == OrientedTargetKind::Tape ?
        estun_vision_grasp::parseVisualTapeTarget(input, now, target, reason) :
        kind == OrientedTargetKind::Pringles ?
        estun_vision_grasp::parseVisualPringlesTarget(input, now, target, reason) :
        estun_vision_grasp::parseVisualBrickTarget(input, now, target, reason);
    if (!parsed)
    {
        RCLCPP_ERROR(logger, "%s：%s", reason.c_str(), path);
        return false;
    }
    return true;
}

bool verifyClosingAxis(const geometry_msgs::msg::Pose& pose,
                       const estun_vision_grasp::VisualBrickTarget& target,
                       const rclcpp::Logger& logger)
{
    Eigen::Quaterniond orientation(pose.orientation.w, pose.orientation.x,
                                   pose.orientation.y, pose.orientation.z);
    if (!orientation.coeffs().allFinite() || orientation.norm() < 1e-9) return false;
    orientation.normalize();
    const Eigen::Matrix3d rotation = orientation.toRotationMatrix();
    const Eigen::Vector3d closing = rotation * Eigen::Vector3d(
        std::cos(estun_vision_grasp::kClosingOffsetRad),
        std::sin(estun_vision_grasp::kClosingOffsetRad), 0.0);
    const Eigen::Vector3d requested(target.short_axis[0], target.short_axis[1], 0.0);
    const double closing_error = std::acos(std::clamp(std::abs(closing.dot(requested)), 0.0, 1.0)) * 180.0 / M_PI;
    const double down_error = std::acos(std::clamp(rotation.col(2).dot(Eigen::Vector3d(0, 0, -1)), -1.0, 1.0)) * 180.0 / M_PI;
    RCLCPP_INFO(logger, "视觉方向复核：夹爪闭合轴与目标轴误差 %.3f deg；朝下误差 %.3f deg", closing_error, down_error);
    if (closing_error > 2.0 || down_error > 2.0)
    {
        RCLCPP_ERROR(logger, "实际抓取朝向误差超限，停止后续动作");
        return false;
    }
    return true;
}


// ============================================================
// 夹爪
// 参数完全沿用已经真机成功版本
// ============================================================

void sendGripper(
    const rclcpp::Publisher<
        step_motor::msg::Motor
    >::SharedPtr& pub,
    bool close)
{
    step_motor::msg::Motor msg;

    msg.id = 1;
    msg.speed = 200;
    msg.dir = close ? 1 : 0;
    msg.mode = 2;
    msg.angle = 30000;
    msg.state = 0;
    msg.sub_divide = 32;

    pub->publish(msg);

    RCLCPP_INFO(
        rclcpp::get_logger("gripper"),
        close
            ? "夹爪：闭合"
            : "夹爪：张开"
    );
}


// ============================================================
// 第一步：给 XYZ，并给定 tool0 orientation
//
// 目的不是执行这条轨迹，
// 只是让 MoveIt 找一组能够把 tool0 放到预抓取点的六轴。
// ============================================================

bool findJointGoalForPosition(
    MoveGroup& group,
    double x,
    double y,
    double z,
    bool align_tool_z_down,
    double acceptable_joint_delta,
    std::array<double, N>& goal,
    const rclcpp::Logger& logger,
    const estun_vision_grasp::VisualBrickTarget* visual_target = nullptr)
{
    // --------------------------------------------------------
    // 当前真实六轴
    // --------------------------------------------------------

    std::array<double, N> current{};
    moveit::core::RobotStatePtr current_state;

    if (!readCurrent(
            group,
            current,
            current_state))
    {
        RCLCPP_ERROR(
            logger,
            "无法读取当前真实六轴"
        );

        return false;
    }

    const Eigen::Quaterniond gripper_ready_orientation = [&]()
    {
        moveit::core::RobotState orientation_state(*current_state);

        for (std::size_t i = 0; i < N; ++i)
        {
            orientation_state.setVariablePosition(
                NAMES[i], estun_vision_grasp::kGripperReady[i]);
        }

        orientation_state.update();

        return Eigen::Quaterniond(
            orientation_state
                .getGlobalLinkTransform("tool0")
                .rotation());
    }();

    // 使用已验证夹爪姿态的水平航向作为搜索起点，但不再
    // 使用它约 18 deg 的倾斜角。仅定位时强制 tool0 +Z
    // 精确对准 base_link -Z，再搜索不同的绕 Z 轴角度。
    const Eigen::Matrix3d ready_rotation =
        gripper_ready_orientation.toRotationMatrix();

    Eigen::Vector3d ready_horizontal_x = ready_rotation.col(0);
    ready_horizontal_x.z() = 0.0;

    const double base_yaw =
        visual_target ? estun_vision_grasp::visualToolYaw(*visual_target) :
        ready_horizontal_x.norm() > 1e-6
            ? std::atan2(ready_horizontal_x.y(), ready_horizontal_x.x())
            : 0.0;

    // 每 30° 覆盖一整圈。相比旧版 20 yaw × 8 IK，当前最多
    // 12 yaw × 4 IK，在保留全局关节距离比较的同时减少搜索量。
    const std::array<double, 12> yaw_offsets_deg = {
          0.0, -30.0, 30.0, -60.0, 60.0,
        -90.0,  90.0, -120.0, 120.0,
       -150.0, 150.0, 180.0
    };

    const auto makeTargetPose =
        [&](double yaw_offset_deg)
        {
            Eigen::Quaterniond orientation =
                gripper_ready_orientation;

            if (align_tool_z_down)
            {
                const double yaw =
                    base_yaw + yaw_offset_deg * M_PI / 180.0;

                const Eigen::Vector3d target_x(
                    std::cos(yaw),
                    std::sin(yaw),
                    0.0
                );
                const Eigen::Vector3d target_z(0.0, 0.0, -1.0);
                const Eigen::Vector3d target_y =
                    target_z.cross(target_x).normalized();

                Eigen::Matrix3d target_rotation;
                target_rotation.col(0) = target_x;
                target_rotation.col(1) = target_y;
                target_rotation.col(2) = target_z;

                orientation = Eigen::Quaterniond(target_rotation);
                orientation.normalize();
            }

            geometry_msgs::msg::Pose pose;
            pose.position.x = x;
            pose.position.y = y;
            pose.position.z = z;
            pose.orientation.x = orientation.x();
            pose.orientation.y = orientation.y();
            pose.orientation.z = orientation.z();
            pose.orientation.w = orientation.w();
            return pose;
        };

    if (align_tool_z_down)
    {
        RCLCPP_INFO(
            logger,
            "%s",
            visual_target
                ? "视觉方向约束：tool0 朝下，夹爪 +45° 闭合轴对齐目标轴；只比较 0°/180° 等价朝向"
                : "仅定位姿态约束：tool0 +Z 严格对准 base_link -Z，比较 12 个 yaw 的可行解"
        );
    }


    // --------------------------------------------------------
    // 固定抓取 orientation 后可能存在多组 IK / 规划终态。
    // 对 12 个 yaw 分别保留最近 IK，最终选择全局最大关节
    // 变化最小的可执行解。
    // --------------------------------------------------------

    constexpr int VERTICAL_YAW_TRIES = 12;
    const int candidate_tries =
        visual_target ? 2 : (align_tool_z_down ? VERTICAL_YAW_TRIES : 1);

    bool found = false;

    double selected_cost =
        std::numeric_limits<double>::infinity();

    std::array<double, N> best_goal{};
    double selected_yaw_offset_deg = 0.0;

    if (align_tool_z_down)
    {
        group.setPlanningTime(1.0);
        group.setNumPlanningAttempts(3);
    }

    const moveit::core::JointModelGroup* joint_model_group =
        current_state->getJointModelGroup("arm");

    if (align_tool_z_down &&
        (!joint_model_group || !joint_model_group->getSolverInstance()))
    {
        RCLCPP_ERROR(
            logger,
            "arm 运动学求解器未加载，无法进行垂直姿态 IK"
        );

        return false;
    }


    for (int attempt = 1;
         attempt <= candidate_tries;
         ++attempt)
    {
        const double yaw_offset_deg =
            visual_target ? (attempt - 1) * 180.0 : align_tool_z_down
                ? yaw_offsets_deg.at(
                    static_cast<std::size_t>(attempt - 1))
                : 0.0;

        const auto target_pose =
            makeTargetPose(yaw_offset_deg);

        group.setStartStateToCurrentState();

        group.clearPoseTargets();

        group.setGoalPositionTolerance(0.005);
        group.setGoalOrientationTolerance(0.02);

        MoveGroup::Plan candidate_plan;

        bool ok = false;
        double yaw_best_cost =
            std::numeric_limits<double>::infinity();
        int yaw_best_ik_try = 0;

        if (align_tool_z_down)
        {
            // 先显式求 IK，再规划到关节目标。直接把严格的 6D
            // 位姿约束交给 OMPL 时，目标采样器可能在规划时限内
            // 一直找不到有效终态，无法区分“无 IK”和“无路径”。
            constexpr int IK_TRIES_PER_YAW = 4;

            for (int ik_try = 1;
                 ik_try <= IK_TRIES_PER_YAW;
                 ++ik_try)
            {
                moveit::core::RobotState ik_state(*current_state);

                // 第一轮使用真实当前位置作为种子，优先寻找邻近解；
                // 后续使用不同种子，避免陷在单一 KDL 分支。
                if (ik_try > 1)
                {
                    ik_state.setToRandomPositions(joint_model_group);
                    ik_state.update();
                }

                if (!ik_state.setFromIK(
                        joint_model_group,
                        target_pose,
                        "tool0",
                        0.10))
                {
                    continue;
                }

                ik_state.update();

                if (!ik_state.satisfiesBounds(joint_model_group))
                {
                    continue;
                }

                group.setStartStateToCurrentState();
                group.clearPoseTargets();

                if (!group.setJointValueTarget(ik_state))
                {
                    continue;
                }

                MoveGroup::Plan ik_plan;

                if (static_cast<bool>(group.plan(ik_plan)))
                {
                    std::array<double, N> ik_goal{};

                    for (std::size_t i = 0; i < N; ++i)
                    {
                        ik_goal[i] =
                            ik_state.getVariablePosition(NAMES[i]);
                    }

                    const double ik_cost =
                        maxDiff(current, ik_goal);

                    if (ik_cost < yaw_best_cost)
                    {
                        ok = true;
                        yaw_best_cost = ik_cost;
                        yaw_best_ik_try = ik_try;
                        candidate_plan = std::move(ik_plan);
                    }
                }
            }

            if (ok)
            {
                RCLCPP_INFO(
                    logger,
                    "候选 %02d：yaw_offset=%+.1f deg，4 次 IK 中第 %d 次关节解最近",
                    attempt,
                    yaw_offset_deg,
                    yaw_best_ik_try
                );
            }
        }
        else
        {
            group.setPoseTarget(target_pose, "tool0");

            ok = static_cast<bool>(
                group.plan(candidate_plan)
            );
        }

        group.clearPoseTargets();

        if (!ok)
        {
            if (align_tool_z_down)
            {
                RCLCPP_WARN(
                    logger,
                    "候选 %02d：yaw_offset=%+.1f deg，4 次 IK/规划均失败",
                    attempt,
                    yaw_offset_deg
                );
            }
            else
            {
                RCLCPP_WARN(
                    logger,
                    "候选 %02d：yaw_offset=%+.1f deg，规划失败",
                    attempt,
                    yaw_offset_deg
                );
            }

            continue;
        }


        const auto& jt =
            candidate_plan
                .trajectory
                .joint_trajectory;

        if (jt.points.empty() ||
            jt.joint_names.size() != N)
        {
            continue;
        }


        std::unordered_map<
            std::string,
            std::size_t
        > index;

        for (std::size_t i = 0;
             i < jt.joint_names.size();
             ++i)
        {
            index[
                jt.joint_names[i]
            ] = i;
        }


        const auto& last =
            jt.points.back();

        std::array<double, N>
            candidate_goal{};

        bool valid = true;

        for (std::size_t i = 0;
             i < N;
             ++i)
        {
            if (!index.count(
                    NAMES[i]))
            {
                valid = false;
                break;
            }

            candidate_goal[i] =
                last.positions[
                    index.at(
                        NAMES[i]
                    )
                ];

            if (!std::isfinite(
                    candidate_goal[i]))
            {
                valid = false;
                break;
            }
        }

        if (!valid)
            continue;


        // ----------------------------------------------------
        // cost:
        // 这里直接使用“最大单关节变化”。
        //
        // 因为后面的 blended 也是按 max joint delta / 0.10
        // 决定需要多少逻辑段。
        // ----------------------------------------------------

        const double cost =
            maxDiff(
                current,
                candidate_goal
            );


        RCLCPP_INFO(
            logger,
            "候选 %02d：yaw_offset=%+.1f deg, max_joint_delta = %.6f rad, "
            "预计至少 %.0f 段",
            attempt,
            yaw_offset_deg,
            cost,
            std::ceil(
                cost /
                JOINT_STEP_MAX
            )
        );


        if (cost < selected_cost)
        {
            found = true;

            selected_cost =
                cost;

            best_goal =
                candidate_goal;

            selected_yaw_offset_deg =
                yaw_offset_deg;

        }
    }

    if (align_tool_z_down)
    {
        group.setPlanningTime(10.0);
        group.setNumPlanningAttempts(10);
    }


    if (!found)
    {
        RCLCPP_ERROR(
            logger,
            "%d 个候选均未找到 IK 和关节路径规划解",
            candidate_tries
        );

        return false;
    }


    goal =
        best_goal;

    RCLCPP_INFO(
        logger,
        "========== 选择全局最近关节解 =========="
    );

    RCLCPP_INFO(
        logger,
        "selected max_joint_delta = %.6f rad",
        selected_cost
    );

    if (align_tool_z_down)
    {
        RCLCPP_INFO(
            logger,
            "选中的垂直向下姿态 yaw_offset = %+.1f deg",
            selected_yaw_offset_deg
        );
    }

    RCLCPP_INFO(
        logger,
        "预计 logical segments = %.0f",
        std::ceil(
            selected_cost /
            JOINT_STEP_MAX
        )
    );

    if (selected_cost > acceptable_joint_delta)
    {
        RCLCPP_ERROR(
            logger,
            "全局最近解仍超过允许关节变化：%.6f > %.6f rad",
            selected_cost,
            acceptable_joint_delta
        );

        return false;
    }


    for (std::size_t i = 0;
         i < N;
         ++i)
    {
        RCLCPP_INFO(
            logger,
            "%s current=% .9f "
            "goal=% .9f "
            "delta=% .9f",
            NAMES[i].c_str(),
            current[i],
            goal[i],
            goal[i] - current[i]
        );
    }


    // --------------------------------------------------------
    // 这里提前限制：
    // 当前程序只接受 MAX_SEGMENTS 个逻辑段以内的解。
    // 避免自动选择一个需要大幅绕转的关节构型。
    // --------------------------------------------------------

    if (selected_cost >
        JOINT_STEP_MAX *
        static_cast<double>(
            MAX_SEGMENTS
        ))
    {
        RCLCPP_ERROR(
            logger,
            "最优候选仍需超过 %zu 段，"
            "拒绝执行",
            MAX_SEGMENTS
        );

        return false;
    }


    return true;
}


// ============================================================
// 从 start 朝 goal 走最多 0.10 rad
// ============================================================

std::array<double, N> nextJointWaypoint(
    const std::array<double, N>& start,
    const std::array<double, N>& goal)
{
    const double error =
        maxDiff(start, goal);

    if (error <= JOINT_STEP_MAX)
        return goal;

    const double ratio =
        JOINT_STEP_MAX / error;

    std::array<double, N> target{};

    for (std::size_t i = 0; i < N; ++i)
    {
        target[i] =
            start[i]
            +
            ratio
            *
            (goal[i] - start[i]);
    }

    return target;
}


// ============================================================
// 单个小关节段检查
// 直接沿用 real_ready_blended_execute_demo 思路
// ============================================================

bool planSmallJointSegment(
    MoveGroup& group,
    moveit::core::RobotStatePtr state,
    const std::array<double, N>& start,
    const std::array<double, N>& target,
    MoveGroup::Plan& plan,
    const rclcpp::Logger& logger)
{
    for (std::size_t i = 0; i < N; ++i)
    {
        state->setVariablePosition(
            NAMES[i],
            start[i]
        );
    }

    state->update();

    group.setStartState(
        *state
    );

    std::vector<double> target_vector(
        target.begin(),
        target.end()
    );

    if (!group.setJointValueTarget(
            target_vector))
    {
        RCLCPP_ERROR(
            logger,
            "setJointValueTarget failed"
        );

        return false;
    }

    plan =
        MoveGroup::Plan();

    if (!group.plan(plan))
    {
        RCLCPP_ERROR(
            logger,
            "小关节段规划失败"
        );

        return false;
    }

    const auto& jt =
        plan.trajectory.joint_trajectory;

    if (jt.points.empty() ||
        jt.joint_names.size() != N)
    {
        return false;
    }

    std::unordered_map<
        std::string,
        std::size_t
    > map;

    for (std::size_t i = 0;
         i < jt.joint_names.size();
         ++i)
    {
        map[jt.joint_names[i]] = i;
    }

    double max_vel = 0.0;
    double max_acc = 0.0;

    for (const auto& p : jt.points)
    {
        if (p.positions.size() != N)
            return false;

        for (std::size_t i = 0; i < N; ++i)
        {
            if (!map.count(NAMES[i]))
                return false;

            const std::size_t j =
                map.at(NAMES[i]);

            if (!std::isfinite(p.positions[j]))
            {
                RCLCPP_ERROR(
                    logger,
                    "小关节段轨迹包含非有限 position"
                );

                return false;
            }

            if (!p.velocities.empty())
            {
                max_vel =
                    std::max(
                        max_vel,
                        std::abs(
                            p.velocities[j]
                        )
                    );
            }

            if (!p.accelerations.empty())
            {
                max_acc =
                    std::max(
                        max_acc,
                        std::abs(
                            p.accelerations[j]
                        )
                    );
            }
        }
    }

    std::array<double, N> final{};

    const auto& last =
        jt.points.back();

    for (std::size_t i = 0; i < N; ++i)
    {
        final[i] =
            last.positions[
                map.at(NAMES[i])
            ];
    }

    const double requested =
        maxDiff(start, target);

    const double final_error =
        maxDiff(final, target);

    RCLCPP_INFO(
        logger,
        "joint segment: "
        "request=%.6f "
        "MoveIt_points=%zu "
        "vel=%.6f "
        "acc=%.6f "
        "final_error=%.6f",
        requested,
        jt.points.size(),
        max_vel,
        max_acc,
        final_error
    );

    if (requested >
        JOINT_STEP_MAX + 1e-9)
    {
        return false;
    }

    if (max_vel >
        MAX_VEL + 1e-9)
    {
        return false;
    }

    if (max_acc >
        MAX_ACC + 1e-9)
    {
        return false;
    }

    if (final_error >
        PLAN_FINAL_TOL)
    {
        return false;
    }

    return true;
}


// ============================================================
// 构造 Current -> Goal 的 blended trajectory
// ============================================================

bool buildBlendedTrajectory(
    MoveGroup& group,
    const std::array<double, N>& current,
    moveit::core::RobotStatePtr state,
    const std::array<double, N>& goal,
    MoveGroup::Plan& combined,
    const std::string& motion_name,
    const rclcpp::Logger& logger)
{
    std::vector<
        std::array<double, N>
    > waypoints;

    std::array<double, N> theoretical =
        current;

    MoveGroup::Plan first_plan;
    bool have_first_plan = false;

    for (std::size_t seg = 0;
         seg < MAX_SEGMENTS;
         ++seg)
    {
        const double error =
            maxDiff(
                theoretical,
                goal
            );

        if (error <=
            JOINT_GOAL_TOL)
        {
            break;
        }

        const auto target =
            nextJointWaypoint(
                theoretical,
                goal
            );

        MoveGroup::Plan plan;

        if (!planSmallJointSegment(
                group,
                state,
                theoretical,
                target,
                plan,
                logger))
        {
            RCLCPP_ERROR(
                logger,
                "Segment %zu 检查失败",
                seg + 1
            );

            return false;
        }

        if (!have_first_plan)
        {
            first_plan = plan;
            have_first_plan = true;
        }

        waypoints.push_back(
            target
        );

        RCLCPP_INFO(
            logger,
            "W%02zu delta=%.6f "
            "q=[%.6f %.6f %.6f %.6f %.6f %.6f]",
            seg + 1,
            maxDiff(
                theoretical,
                target
            ),
            target[0],
            target[1],
            target[2],
            target[3],
            target[4],
            target[5]
        );

        theoretical =
            target;
    }

    if (!have_first_plan ||
        waypoints.empty())
    {
        RCLCPP_ERROR(
            logger,
            "没有生成有效 joint waypoint"
        );

        return false;
    }

    if (maxDiff(
            theoretical,
            goal)
        >
        JOINT_GOAL_TOL)
    {
        RCLCPP_ERROR(
            logger,
            "MAX_SEGMENTS=%zu 不足，"
            "目标关节变化过大",
            MAX_SEGMENTS
        );

        return false;
    }

    combined =
        MoveGroup::Plan();

    combined.start_state =
        first_plan.start_state;

    auto& jt =
        combined
            .trajectory
            .joint_trajectory;

    jt.joint_names =
        first_plan
            .trajectory
            .joint_trajectory
            .joint_names;

    std::unordered_map<
        std::string,
        std::size_t
    > canonical;

    for (std::size_t i = 0; i < N; ++i)
    {
        canonical[
            NAMES[i]
        ] = i;
    }

    auto makePoint =
        [&](const std::array<double, N>& q,
            int sec)
        {
            decltype(jt.points)::value_type p;

            for (const auto& name :
                 jt.joint_names)
            {
                if (!canonical.count(name))
                {
                    throw std::runtime_error(
                        "unexpected joint name"
                    );
                }

                p.positions.push_back(
                    q[
                        canonical.at(name)
                    ]
                );
            }

            // real_robot_bridge 会使用自己的
            // REAL_VEL / REAL_ACC
            p.velocities.assign(
                N,
                0.0
            );

            p.accelerations.assign(
                N,
                0.0
            );

            p.time_from_start.sec =
                sec;

            p.time_from_start.nanosec =
                0;

            return p;
        };

    try
    {
        // 第一点为真实 Current
        jt.points.push_back(
            makePoint(
                current,
                0
            )
        );

        for (std::size_t i = 0;
             i < waypoints.size();
             ++i)
        {
            jt.points.push_back(
                makePoint(
                    waypoints[i],
                    static_cast<int>(
                        i + 1
                    )
                )
            );
        }
    }
    catch (const std::exception& e)
    {
        RCLCPP_ERROR(
            logger,
            "构造 trajectory 失败: %s",
            e.what()
        );

        return false;
    }

    RCLCPP_INFO(
        logger,
        "========== %s SUMMARY ==========",
        motion_name.c_str()
    );

    RCLCPP_INFO(
        logger,
        "逻辑运动段数 = %zu",
        waypoints.size()
    );

    RCLCPP_INFO(
        logger,
        "发送 trajectory points = %zu",
        jt.points.size()
    );

    RCLCPP_INFO(
        logger,
        "最大 joint 分段 = %.3f rad",
        JOINT_STEP_MAX
    );

    RCLCPP_INFO(
        logger,
        "最终 joint 目标误差 = %.9f",
        maxDiff(
            waypoints.back(),
            goal
        )
    );

    return true;
}


// ============================================================
// Cartesian 轨迹安全检查
// 沿用 real_cartesian_down_up
// ============================================================

bool cartesianSafetyCheck(
    const moveit_msgs::msg::RobotTrajectory& trajectory,
    const moveit::core::RobotStatePtr& start_state,
    const rclcpp::Logger& logger)
{
    const auto& jt =
        trajectory.joint_trajectory;

    if (jt.joint_names.empty() ||
        jt.points.empty())
    {
        return false;
    }

    const std::size_t n =
        jt.joint_names.size();

    double max_delta = 0.0;
    double max_vel = 0.0;
    double max_acc = 0.0;

    for (const auto& point :
         jt.points)
    {
        if (point.positions.size() != n)
            return false;

        if ((!point.velocities.empty() &&
             point.velocities.size() != n) ||
            (!point.accelerations.empty() &&
             point.accelerations.size() != n))
        {
            return false;
        }

        for (std::size_t j = 0;
             j < n;
             ++j)
        {
            const auto& name =
                jt.joint_names[j];

            const double current =
                start_state
                    ->getVariablePosition(
                        name
                    );

            if (!std::isfinite(current) ||
                !std::isfinite(point.positions[j]))
            {
                return false;
            }

            max_delta =
                std::max(
                    max_delta,
                    std::abs(
                        point.positions[j]
                        -
                        current
                    )
                );

            if (!point.velocities.empty())
            {
                if (!std::isfinite(point.velocities[j]))
                    return false;

                max_vel =
                    std::max(
                        max_vel,
                        std::abs(
                            point.velocities[j]
                        )
                    );
            }

            if (!point.accelerations.empty())
            {
                if (!std::isfinite(point.accelerations[j]))
                    return false;

                max_acc =
                    std::max(
                        max_acc,
                        std::abs(
                            point.accelerations[j]
                        )
                    );
            }
        }
    }

    RCLCPP_INFO(
        logger,
        "Cartesian 安全检查: "
        "delta=%.6f vel=%.6f acc=%.6f",
        max_delta,
        max_vel,
        max_acc
    );

    if (max_delta >
        CART_MAX_DELTA)
        return false;

    if (max_vel >
        CART_MAX_VEL + 0.001)
        return false;

    if (max_acc >
        CART_MAX_ACC + 0.001)
        return false;

    return true;
}


// ============================================================
// 单个 Cartesian 小段
// ============================================================

bool executeCartesianTarget(
    MoveGroup& group,
    const geometry_msgs::msg::Pose& target_pose,
    const std::string& label,
    const rclcpp::Logger& logger)
{
    group.setStartStateToCurrentState();

    auto start_state =
        group.getCurrentState(5.0);

    if (!start_state)
    {
        RCLCPP_ERROR(
            logger,
            "%s：无法获取 RobotState",
            label.c_str()
        );

        return false;
    }

    std::vector<
        geometry_msgs::msg::Pose
    > waypoints{
        target_pose
    };

    moveit_msgs::msg::
        RobotTrajectory trajectory;

    moveit_msgs::msg::
        MoveItErrorCodes error_code;

    const double fraction =
        group.computeCartesianPath(
            waypoints,
            EEF_STEP,
            trajectory,
            true,
            &error_code
        );

    RCLCPP_INFO(
        logger,
        "%s：Cartesian fraction = %.6f",
        label.c_str(),
        fraction
    );

    RCLCPP_INFO(
        logger,
        "%s：trajectory points = %zu",
        label.c_str(),
        trajectory
            .joint_trajectory
            .points
            .size()
    );

    if (fraction <
        FRACTION_REQUIRED)
    {
        RCLCPP_ERROR(
            logger,
            "%s：Cartesian 路径不完整",
            label.c_str()
        );

        return false;
    }

    if (!cartesianSafetyCheck(
            trajectory,
            start_state,
            logger))
    {
        RCLCPP_ERROR(
            logger,
            "%s：安全检查失败",
            label.c_str()
        );

        return false;
    }

    // real_robot_bridge 使用轨迹中高于默认值的速度/加速度作为
    // 本次 RobotMove 的显式请求，并继续受桥接层 0.20 上限保护。
    // 这一步放在 MoveIt 原始 Cartesian 轨迹安全检查之后，避免
    // 把协议速度提示混入 MoveIt 规划质量检查。
    const std::size_t joint_count =
        trajectory.joint_trajectory.joint_names.size();

    for (auto& point : trajectory.joint_trajectory.points)
    {
        point.velocities.assign(joint_count, CART_REAL_VEL);
        point.accelerations.assign(joint_count, CART_REAL_ACC);
    }

    MoveGroup::Plan plan;
    plan.trajectory =
        trajectory;

    const auto result =
        group.execute(plan);

    if (result !=
        moveit::core::
            MoveItErrorCode::SUCCESS)
    {
        RCLCPP_ERROR(
            logger,
            "%s：Execute 失败",
            label.c_str()
        );

        return false;
    }

    std::this_thread::sleep_for(100ms);

    // 每个 20 mm 小段执行完成后，立即核对真实关节终点和 TCP。
    // 任一小段没有实际到位，后续下降/上升段都不得继续。
    auto actual_state =
        group.getCurrentState(5.0);

    if (!actual_state)
    {
        RCLCPP_ERROR(
            logger,
            "%s：执行后无法读取真实关节状态",
            label.c_str()
        );

        return false;
    }

    const auto& planned_last =
        trajectory.joint_trajectory.points.back();

    double actual_joint_error = 0.0;

    for (std::size_t j = 0;
         j < trajectory.joint_trajectory.joint_names.size();
         ++j)
    {
        const double actual =
            actual_state->getVariablePosition(
                trajectory.joint_trajectory.joint_names[j]
            );

        if (!std::isfinite(actual))
        {
            RCLCPP_ERROR(
                logger,
                "%s：执行后真实关节状态包含非有限数",
                label.c_str()
            );

            return false;
        }

        actual_joint_error = std::max(
            actual_joint_error,
            std::abs(actual - planned_last.positions[j])
        );
    }

    const auto actual_pose =
        group.getCurrentPose("tool0").pose;

    const double dx = actual_pose.position.x - target_pose.position.x;
    const double dy = actual_pose.position.y - target_pose.position.y;
    const double dz = actual_pose.position.z - target_pose.position.z;
    const double actual_position_error =
        std::sqrt(dx * dx + dy * dy + dz * dz);

    Eigen::Quaterniond actual_q(
        actual_pose.orientation.w,
        actual_pose.orientation.x,
        actual_pose.orientation.y,
        actual_pose.orientation.z
    );
    Eigen::Quaterniond target_q(
        target_pose.orientation.w,
        target_pose.orientation.x,
        target_pose.orientation.y,
        target_pose.orientation.z
    );

    actual_q.normalize();
    target_q.normalize();

    const double actual_orientation_error_deg =
        actual_q.angularDistance(target_q) * 180.0 / M_PI;

    RCLCPP_INFO(
        logger,
        "%s：执行后 joint_error=%.6f rad，TCP position_error=%.3f mm，orientation_error=%.3f deg",
        label.c_str(),
        actual_joint_error,
        actual_position_error * 1000.0,
        actual_orientation_error_deg
    );

    if (actual_joint_error > FINAL_ACTUAL_TOL ||
        actual_position_error > 0.010 ||
        actual_orientation_error_deg > 2.0)
    {
        RCLCPP_ERROR(
            logger,
            "%s：实际终点核对失败，停止后续 Cartesian 小段",
            label.c_str()
        );

        return false;
    }

    return true;
}


// ============================================================
// 沿 base_link Z 方向分段 Cartesian 运动
// 每段最多 20 mm
// ============================================================

bool moveVertical(
    MoveGroup& group,
    double distance,
    const std::string& motion_name,
    const rclcpp::Logger& logger)
{
    double remaining =
        std::abs(distance);

    const double direction =
        distance >= 0.0 ? 1.0 : -1.0;

    int seg = 1;

    while (remaining > 1e-6)
    {
        const double dz = direction *
            std::min(
                CART_SEGMENT,
                remaining
            );

        const auto current =
            group
                .getCurrentPose(
                    "tool0"
                )
                .pose;

        auto target =
            current;

        target.position.z +=
            dz;

        const std::string label =
            motion_name + "段 "
            +
            std::to_string(seg);

        RCLCPP_INFO(
            logger,
            "%s：base_link Z %+.1f mm",
            label.c_str(),
            dz * 1000.0
        );

        if (!executeCartesianTarget(
                group,
                target,
                label,
                logger))
        {
            return false;
        }

        remaining -= std::abs(dz);
        ++seg;

        // executeCartesianTarget 已等待真机完成、等待状态刷新并核对
        // 实际关节/TCP；不再额外停顿 300 ms。
    }

    return true;
}

} // namespace


// ============================================================
// main
// ============================================================

int main(
    int argc,
    char** argv)
{
    rclcpp::init(
        argc,
        argv
    );

    auto node =
        rclcpp::Node::make_shared(
            "foam_brick_grasp",
            rclcpp::NodeOptions()
                .automatically_declare_parameters_from_overrides(
                    true
                )
        );

    const auto logger =
        node->get_logger();

    // MoveGroupInterface 会在本进程中构造 RobotModel。显式提供
    // kinematics 参数，确保本地 RobotState 可以执行 setFromIK；
    // 规划服务器仍使用 MoveIt config 中的同一套 KDL 配置。
    if (!node->has_parameter(
            "robot_description_kinematics.arm.kinematics_solver"))
    {
        node->declare_parameter(
            "robot_description_kinematics.arm.kinematics_solver",
            "kdl_kinematics_plugin/KDLKinematicsPlugin"
        );
    }

    if (!node->has_parameter(
            "robot_description_kinematics.arm.kinematics_solver_search_resolution"))
    {
        node->declare_parameter(
            "robot_description_kinematics.arm.kinematics_solver_search_resolution",
            0.005
        );
    }

    if (!node->has_parameter(
            "robot_description_kinematics.arm.kinematics_solver_timeout"))
    {
        node->declare_parameter(
            "robot_description_kinematics.arm.kinematics_solver_timeout",
            0.10
        );
    }

    if (!node->has_parameter(
            "robot_description_kinematics.arm.kinematics_solver_attempts"))
    {
        node->declare_parameter(
            "robot_description_kinematics.arm.kinematics_solver_attempts",
            3
        );
    }

    bool positioning_only = false;
    bool bare_flange_positioning = false;
    bool execute_positioning = false;
    bool use_visual_orientation = false;
    bool use_tape_outer_orientation = false;
    bool use_pringles_orientation = false;
    bool execute_visual_grasp = false;
    double positioning_clearance = 0.30;
    double positioning_max_joint_delta = 1.00;

    node->get_parameter_or(
        "positioning_only",
        positioning_only,
        false
    );

    node->get_parameter_or(
        "bare_flange_positioning",
        bare_flange_positioning,
        false
    );

    node->get_parameter_or(
        "execute_positioning",
        execute_positioning,
        false
    );

    node->get_parameter_or("use_visual_orientation", use_visual_orientation, false);
    node->get_parameter_or("use_tape_outer_orientation", use_tape_outer_orientation, false);
    node->get_parameter_or("use_pringles_orientation", use_pringles_orientation, false);
    node->get_parameter_or("execute_visual_grasp", execute_visual_grasp, false);
    const bool oriented_mode = use_visual_orientation || use_tape_outer_orientation ||
                               use_pringles_orientation;
    const OrientedTargetKind oriented_kind = use_tape_outer_orientation ? OrientedTargetKind::Tape :
                                             use_pringles_orientation ? OrientedTargetKind::Pringles :
                                                                         OrientedTargetKind::Brick;
    if ((oriented_mode && bare_flange_positioning) ||
        static_cast<int>(use_visual_orientation) + static_cast<int>(use_tape_outer_orientation) +
            static_cast<int>(use_pringles_orientation) > 1)
    {
        RCLCPP_ERROR(logger, "视觉方向模式不能与裸法兰模式或另一种视觉方向模式同时启用");
        rclcpp::shutdown();
        return 1;
    }

    node->get_parameter_or(
        "positioning_clearance",
        positioning_clearance,
        bare_flange_positioning ? 0.15 : 0.30
    );

    node->get_parameter_or(
        "positioning_max_joint_delta",
        positioning_max_joint_delta,
        1.00
    );

    if (bare_flange_positioning && !positioning_only)
    {
        RCLCPP_ERROR(logger, "裸法兰模式必须同时设置 positioning_only:=true");
        rclcpp::shutdown();
        return 1;
    }

    if (positioning_only &&
        (positioning_clearance < 0.10 ||
         positioning_clearance > 0.58))
    {
        RCLCPP_ERROR(
            logger,
            "positioning_clearance 必须在 0.10～0.58 m，当前为 %.3f m",
            positioning_clearance
        );

        rclcpp::shutdown();
        return 1;
    }

    if (positioning_only &&
        (positioning_max_joint_delta < 0.10 ||
         positioning_max_joint_delta > 2.00))
    {
        RCLCPP_ERROR(
            logger,
            "positioning_max_joint_delta 必须在 0.10～2.00 rad，当前为 %.3f rad",
            positioning_max_joint_delta
        );

        rclcpp::shutdown();
        return 1;
    }

    rclcpp::executors::
        MultiThreadedExecutor executor;

    executor.add_node(node);

    std::thread spin(
        [&executor]()
        {
            executor.spin();
        }
    );

    auto finish =
        [&](int code)
        {
            executor.cancel();

            if (spin.joinable())
                spin.join();

            rclcpp::shutdown();

            return code;
        };

    std::mutex robot_info_mutex;
    std::string robot_mode;
    int robot_fault = -1;
    std::chrono::steady_clock::time_point robot_info_time{};
    bool robot_info_received = false;
    auto robot_info_subscription = node->create_subscription<codroid_msgs::msg::RobotInfo>(
        "/RobotInfo", 10,
        [&](const codroid_msgs::msg::RobotInfo::SharedPtr message)
        {
            std::lock_guard<std::mutex> lock(robot_info_mutex);
            robot_mode = message->state;
            robot_fault = message->fault_flag;
            robot_info_time = std::chrono::steady_clock::now();
            robot_info_received = true;
        });


    MoveGroup group(
        node,
        "arm"
    );

    group.setPoseReferenceFrame(
        "base_link"
    );

    group.setEndEffectorLink(
        "tool0"
    );

    group.setPlanningTime(
        10.0
    );

    group.setNumPlanningAttempts(
        10
    );

    // 与已经真机成功代码统一
    group.setMaxVelocityScalingFactor(
        0.03
    );

    group.setMaxAccelerationScalingFactor(
        0.03
    );


    auto gripper_pub =
        node->create_publisher<
            step_motor::msg::Motor
        >(
            "/motor_control",
            10
        );

    std::this_thread::sleep_for(
        1s
    );


    // ---------------------------------------------------------
    // 视觉目标
    // ---------------------------------------------------------

    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double confidence = 0.0;
    estun_vision_grasp::VisualBrickTarget visual_target;

    if (oriented_mode)
    {
        if (!readVisualTarget(visual_target, logger, oriented_kind)) return finish(1);
        x = visual_target.xyz[0];
        y = visual_target.xyz[1];
        z = visual_target.xyz[2];
        confidence = visual_target.confidence;
        RCLCPP_INFO(logger, "%s = [%.6f %.6f]，夹爪安装角 +45°，目标 tool0 X 水平角 %.2f deg（及180°等价解）",
                    use_tape_outer_orientation ? "胶带径向闭合轴" :
                    use_pringles_orientation ? "薯片筒横向闭合轴" : "砖块短轴",
                    visual_target.short_axis[0], visual_target.short_axis[1],
                    estun_vision_grasp::visualToolYaw(visual_target) * 180.0 / M_PI);
    }
    else if (!readTarget(
            x,
            y,
            z,
            confidence))
    {
        return finish(1);
    }


    // ---------------------------------------------------------
    // 当前真实六轴
    // ---------------------------------------------------------

    std::array<double, N>
        current{};

    moveit::core::
        RobotStatePtr state;

    if (!readCurrent(
            group,
            current,
            state))
    {
        RCLCPP_ERROR(
            logger,
            "无法读取当前六轴"
        );

        return finish(1);
    }

    double flange_to_tool0_m = 0.0;
    if (bare_flange_positioning)
    {
        // 模型中的 tool0 是虚拟 TCP；无夹爪时验收点是实体法兰 link6。
        const Eigen::Isometry3d flange_to_tool0 =
            state->getGlobalLinkTransform("link6").inverse() *
            state->getGlobalLinkTransform("tool0");
        const Eigen::Vector3d offset = flange_to_tool0.translation();
        if (!offset.allFinite() ||
            !flange_to_tool0.rotation().isApprox(Eigen::Matrix3d::Identity(), 1e-6) ||
            std::abs(offset.x()) > 1e-6 ||
            std::abs(offset.y()) > 1e-6 ||
            offset.z() < 0.05 || offset.z() > 0.30)
        {
            RCLCPP_ERROR(logger, "link6 -> tool0 模型变换异常，拒绝裸法兰定位");
            return finish(1);
        }
        flange_to_tool0_m = offset.z();
        RCLCPP_INFO(logger, "裸法兰模式：模型 link6 -> tool0 = %.1f mm", 1000.0 * flange_to_tool0_m);
    }


    // ---------------------------------------------------------
    // 先只规划：
    // 找到能够把 TCP 放到砖块上方预抓取点的最终六轴。
    // ---------------------------------------------------------

    std::array<double, N>
        goal{};

    const double approach_clearance =
        positioning_only
            ? positioning_clearance
            : APPROACH_DISTANCE;

    const double approach_z =
        z + approach_clearance -
        (bare_flange_positioning ? flange_to_tool0_m : 0.0);

    const double acceptable_joint_delta =
        positioning_only
            ? positioning_max_joint_delta
            : JOINT_STEP_MAX * static_cast<double>(MAX_SEGMENTS);

    if (!findJointGoalForPosition(
            group,
            x,
            y,
            approach_z,
            true,
            acceptable_joint_delta,
            goal,
            logger,
            oriented_mode ? &visual_target : nullptr))
    {
        return finish(2);
    }

    if (positioning_only)
    {
        const double positioning_joint_delta =
            maxDiff(current, goal);

        if (positioning_joint_delta > positioning_max_joint_delta)
        {
            RCLCPP_ERROR(
                logger,
                "仅定位目标关节变化过大：%.6f > %.6f rad，拒绝执行",
                positioning_joint_delta,
                positioning_max_joint_delta
            );

            return finish(2);
        }
    }


    // ---------------------------------------------------------
    // 按 0.10 rad 分段检查并组合 trajectory
    // ---------------------------------------------------------

    MoveGroup::Plan combined;

    if (!buildBlendedTrajectory(
            group,
            current,
            state,
            goal,
            combined,
            "BRICK APPROACH",
            logger))
    {
        return finish(3);
    }


    std::cout
        << "\n========================================\n"
        << (use_tape_outer_orientation ? "Tape outer rim\n" :
            use_pringles_orientation ? "Pringles body\n" : "Foam Brick\n")
        << "confidence = "
        << confidence
        << "\n"
        << "base_link XYZ = ["
        << x << ", "
        << y << ", "
        << z << "] m\n"
        << "\n位置目标已找到可达六轴。\n"
        << "所有 0.10 rad 小段已经通过规划检查。\n";

    if (oriented_mode)
    {
        std::cout << (use_tape_outer_orientation ?
            "胶带外圈模式：夹爪闭合轴穿过胶带中心与外圈落点；不会退回任意 yaw 候选。\n" :
            use_pringles_orientation ?
            "薯片筒模式：夹爪闭合轴垂直筒身长轴；不会退回任意 yaw 候选。\n" :
            "视觉方向模式：夹爪闭合轴沿砖块短轴；不会退回任意 yaw 候选。\n");
        if (!execute_visual_grasp && !positioning_only)
        {
            RCLCPP_WARN(logger, "视觉方向 PLAN-ONLY 完成；执行需 execute_visual_grasp:=true，并输入确认口令");
            return finish(0);
        }
    }

    if (positioning_only)
    {
        std::cout
            << "\n当前为仅定位模式：\n"
            << "1. 不发送任何夹爪命令\n"
            << "2. 一次连续 Execute，"
            << (bare_flange_positioning ? "裸法兰 link6" : "tool0")
            << (use_tape_outer_orientation ? " 到胶带外圈点上方 " :
                use_pringles_orientation ? " 到薯片筒中心上方 " : " 到砖块检测点上方 ")
            << positioning_clearance * 1000.0
            << " mm\n"
            << "3. 到达后立即退出，不下降、不抬升\n";
        if (bare_flange_positioning && !execute_positioning)
        {
            RCLCPP_WARN(logger, "裸法兰 PLAN-ONLY 完成；需显式 execute_positioning:=true 才会请求执行");
            return finish(0);
        }
        std::cout << "\n准确输入 POSITION 才执行：";
    }
    else
    {
        std::cout
            << "\n执行流程：\n"
            << (use_tape_outer_orientation ?
                "1. 一次连续 Execute 到胶带左侧外圈点上方 150 mm\n" :
                use_pringles_orientation ?
                "1. 一次连续 Execute 到薯片筒中段上方 150 mm\n" :
                "1. 一次连续 Execute 到砖块上方 150 mm\n")
            << "2. 夹爪张开\n"
            << "3. 沿 base_link -Z 每段最多 20 mm，下降 150 mm\n"
            << (use_tape_outer_orientation ?
                "4. 到达胶带左侧实体带后夹爪闭合\n" :
                use_pringles_orientation ?
                "4. 到达薯片筒中段后夹爪闭合\n" :
                "4. 到达砖块中心后夹爪闭合\n")
            << "5. 沿 base_link +Z 每段最多 20 mm，上升 150 mm\n"
            << "6. 一次连续 Execute 返回 Gripper Ready\n"
            << "\n准确输入 GRASP 才执行：";
    }

    std::cout << std::flush;


    std::string answer;
    std::cin >> answer;

    const std::string expected_answer =
        positioning_only
            ? "POSITION"
            : "GRASP";

    if (answer != expected_answer)
    {
        RCLCPP_WARN(
            logger,
            "用户取消，不运动"
        );

        return finish(0);
    }

    if (oriented_mode)
    {
        estun_vision_grasp::VisualBrickTarget latest;
        if (!readVisualTarget(latest, logger, oriented_kind) ||
            !estun_vision_grasp::sameVisualBrickTarget(visual_target, latest))
        {
            RCLCPP_ERROR(logger, "确认期间视觉目标失效、移动超过10 mm或转动超过5°，旧规划作废");
            return finish(4);
        }
    }
    else if (bare_flange_positioning)
    {
        double latest_x = 0.0;
        double latest_y = 0.0;
        double latest_z = 0.0;
        double latest_confidence = 0.0;
        if (!readTarget(latest_x, latest_y, latest_z, latest_confidence) ||
            std::abs(latest_x - x) > 0.010 ||
            std::abs(latest_y - y) > 0.010 ||
            std::abs(latest_z - z) > 0.010)
        {
            RCLCPP_ERROR(logger, "确认期间视觉目标失效或移动超过 10 mm，旧规划作废");
            return finish(4);
        }
    }


    // =========================================================
    // 与 return_to_gripper_ready 一致：
    // 执行前重新读取真实 Current，防止确认或等待期间旧规划失效。
    // =========================================================

    std::array<double, N>
        before_execute{};

    moveit::core::
        RobotStatePtr before_execute_state;

    if (!readCurrent(
            group,
            before_execute,
            before_execute_state))
    {
        RCLCPP_ERROR(
            logger,
            "执行前无法重新读取真实六轴状态"
        );

        return finish(4);
    }

    const double start_change =
        maxDiff(
            current,
            before_execute
        );

    RCLCPP_INFO(
        logger,
        "确认后真实起点变化 = %.9f rad",
        start_change
    );

    if (start_change >
        START_STATE_TOL)
    {
        RCLCPP_ERROR(
            logger,
            "确认期间机械臂发生移动：%.6f > %.6f rad，旧规划作废",
            start_change,
            START_STATE_TOL
        );

        return finish(4);
    }

    if (bare_flange_positioning || oriented_mode)
    {
        std::string mode_snapshot;
        int fault_snapshot = -1;
        double info_age = std::numeric_limits<double>::infinity();
        bool received_snapshot = false;
        {
            std::lock_guard<std::mutex> lock(robot_info_mutex);
            received_snapshot = robot_info_received;
            mode_snapshot = robot_mode;
            fault_snapshot = robot_fault;
            if (robot_info_received)
                info_age = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - robot_info_time).count();
        }
        if (!received_snapshot || info_age > 2.0 ||
            mode_snapshot != "AutoReady" || fault_snapshot != 0)
        {
            RCLCPP_ERROR(logger,
                "真机未就绪：/RobotInfo state=%s fault_flag=%d age=%.2f s；需要 AutoReady 且 fault_flag=0",
                received_snapshot ? mode_snapshot.c_str() : "未收到",
                fault_snapshot, info_age);
            return finish(4);
        }
        RCLCPP_INFO(logger, "真机状态已复核：AutoReady，fault_flag=0");
    }


    // =========================================================
    // 一次连续 blended Execute 到预抓取点
    // =========================================================

    RCLCPP_WARN(
        logger,
        "开始执行到 Foam Brick Approach"
    );

    const auto move_result =
        group.execute(
            combined
        );

    if (!move_result)
    {
        RCLCPP_ERROR(
            logger,
            "到预抓取点 Execute 失败"
        );

        return finish(4);
    }


    std::this_thread::sleep_for(
        300ms
    );


    // ---------------------------------------------------------
    // 检查实际 joint 是否到了位置目标
    // ---------------------------------------------------------

    std::array<double, N>
        actual{};

    moveit::core::
        RobotStatePtr final_state;

    if (!readCurrent(
            group,
            actual,
            final_state))
    {
        RCLCPP_ERROR(
            logger,
            "执行后无法读取实际状态"
        );

        return finish(5);
    }

    const double joint_error =
        maxDiff(
            actual,
            goal
        );

    RCLCPP_INFO(
        logger,
        "到砖块后最大 joint error = %.9f rad",
        joint_error
    );

    if (joint_error >
        FINAL_ACTUAL_TOL)
    {
        RCLCPP_ERROR(
            logger,
            "实际位置误差过大，"
            "不闭合夹爪"
        );

        return finish(6);
    }


    const auto actual_pose =
        group
            .getCurrentPose(
                "tool0"
            )
            .pose;

    RCLCPP_INFO(
        logger,
        "到达后实际 tool0 = "
        "[%.6f %.6f %.6f]",
        actual_pose.position.x,
        actual_pose.position.y,
        actual_pose.position.z
    );

    const double dx =
        actual_pose.position.x - x;
    const double dy =
        actual_pose.position.y - y;
    const double dz =
        actual_pose.position.z - approach_z;

    const double position_error =
        std::sqrt(dx * dx + dy * dy + dz * dz);

    RCLCPP_INFO(
        logger,
        "预抓取目标 = [%.6f %.6f %.6f]",
        x, y, approach_z
    );

    RCLCPP_INFO(
        logger,
        "TCP 到预抓取点位置误差 = %.3f mm",
        position_error * 1000.0
    );

    if (position_error > 0.010)
    {
        RCLCPP_ERROR(
            logger,
            "预抓取点误差超过 10 mm，不下降"
        );

        return finish(7);
    }

    if (oriented_mode && !verifyClosingAxis(actual_pose, visual_target, logger))
        return finish(7);

    if (positioning_only)
    {
        if (bare_flange_positioning)
        {
            const auto flange_pose = group.getCurrentPose("link6").pose;
            const double flange_error = std::sqrt(
                std::pow(flange_pose.position.x - x, 2) +
                std::pow(flange_pose.position.y - y, 2) +
                std::pow(flange_pose.position.z - (z + positioning_clearance), 2));
            RCLCPP_INFO(logger,
                "实际裸法兰 link6 = [%.6f %.6f %.6f] m；目标 = [%.6f %.6f %.6f] m；误差 = %.3f mm",
                flange_pose.position.x, flange_pose.position.y, flange_pose.position.z,
                x, y, z + positioning_clearance, flange_error * 1000.0);
            if (flange_error > 0.010)
            {
                RCLCPP_ERROR(logger, "裸法兰到目标上方的误差超过 10 mm");
                return finish(7);
            }
        }
        Eigen::Quaterniond actual_orientation(
            actual_pose.orientation.w,
            actual_pose.orientation.x,
            actual_pose.orientation.y,
            actual_pose.orientation.z
        );
        actual_orientation.normalize();

        const Eigen::Vector3d actual_tool_z =
            actual_orientation.toRotationMatrix().col(2);

        const double downward_alignment = std::clamp(
            actual_tool_z.dot(Eigen::Vector3d(0.0, 0.0, -1.0)),
            -1.0,
            1.0
        );

        const double axis_error_deg =
            std::acos(downward_alignment) * 180.0 / M_PI;

        RCLCPP_INFO(
            logger,
            "tool0 +Z 对准砖块方向的角度误差 = %.3f deg",
            axis_error_deg
        );

        if (axis_error_deg > 2.0)
        {
            RCLCPP_ERROR(
                logger,
                "tool0 Z 轴朝向误差超过 2 deg，仅定位验证失败"
            );

            return finish(7);
        }

        RCLCPP_INFO(
            logger,
            "仅定位完成：%s 已到目标点上方 %.1f mm，+Z 轴已朝下；不下降、不发送夹爪命令",
            bare_flange_positioning ? "裸法兰 link6" : "tool0",
            positioning_clearance * 1000.0
        );

        return finish(0);
    }

    // =========================================================
    // 已确认到达砖块上方 150 mm 后再张开夹爪
    // =========================================================

    sendGripper(
        gripper_pub,
        false
    );

    std::this_thread::sleep_for(
        2s
    );

    // =========================================================
    // 从预抓取点沿 base_link -Z 下降到抓取点
    // =========================================================

    RCLCPP_INFO(
        logger,
        "%s", use_tape_outer_orientation ? "开始下降到胶带外圈抓取点" : "开始下降到泡沫砖抓取点"
    );

    if (!moveVertical(
            group,
            -APPROACH_DISTANCE,
            "下降",
            logger))
    {
        RCLCPP_ERROR(
            logger,
            "下降到抓取点失败，不闭合夹爪"
        );

        return finish(8);
    }

    const auto grasp_pose =
        group.getCurrentPose("tool0").pose;

    if (oriented_mode && !verifyClosingAxis(grasp_pose, visual_target, logger))
        return finish(9);

    const double grasp_dx = grasp_pose.position.x - x;
    const double grasp_dy = grasp_pose.position.y - y;
    const double grasp_dz = grasp_pose.position.z - z;
    const double grasp_error = std::sqrt(
        grasp_dx * grasp_dx +
        grasp_dy * grasp_dy +
        grasp_dz * grasp_dz);

    RCLCPP_INFO(
        logger,
        "抓取点 TCP 位置误差 = %.3f mm",
        grasp_error * 1000.0
    );

    if (grasp_error > 0.010)
    {
        RCLCPP_ERROR(
            logger,
            "抓取点误差超过 10 mm，不闭合夹爪"
        );

        return finish(9);
    }

    // =========================================================
    // 闭合
    // =========================================================

    sendGripper(
        gripper_pub,
        true
    );

    std::this_thread::sleep_for(
        2s
    );


    // =========================================================
    // 沿原路上升 150 mm
    // =========================================================

    RCLCPP_INFO(
        logger,
        "开始沿 base_link +Z 抬升泡沫砖 150 mm"
    );

    if (!moveVertical(
            group,
            LIFT_DISTANCE,
            "抬升",
            logger))
    {
        RCLCPP_ERROR(
            logger,
            "砖块已经夹住，"
            "但自动抬升失败"
        );

        return finish(10);
    }

    const auto lifted_pose =
        group.getCurrentPose("tool0").pose;

    const double lifted_dx = lifted_pose.position.x - x;
    const double lifted_dy = lifted_pose.position.y - y;
    const double lifted_dz = lifted_pose.position.z - approach_z;
    const double lifted_error = std::sqrt(
        lifted_dx * lifted_dx +
        lifted_dy * lifted_dy +
        lifted_dz * lifted_dz);

    RCLCPP_INFO(
        logger,
        "抬升后 TCP 到预抓取点误差 = %.3f mm",
        lifted_error * 1000.0
    );

    if (lifted_error > 0.010)
    {
        RCLCPP_ERROR(
            logger,
            "砖块已夹住，但抬升终点误差超过 10 mm；保持夹紧并停止"
        );

        return finish(11);
    }

    // =========================================================
    // 抬升完成后返回 Gripper Ready。
    // 与已通过真机验证的 return_to_gripper_ready 完全相同：
    // 0.10 rad 逻辑分段逐段规划检查，丢弃 MoveIt 稠密中间点，
    // 最终仅发送 Current + W01...Wnn，一次连续 Execute。
    // =========================================================

    std::array<double, N> ready_start{};
    moveit::core::RobotStatePtr ready_state;

    if (!readCurrent(
            group,
            ready_start,
            ready_state))
    {
        RCLCPP_ERROR(
            logger,
            "砖块已夹住，但无法读取回 Ready 前的真实六轴；保持夹紧并停止"
        );

        return finish(12);
    }

    const auto& ready_goal =
        estun_vision_grasp::kGripperReady;

    const double ready_delta =
        maxDiff(ready_start, ready_goal);

    RCLCPP_INFO(
        logger,
        "抬升位置 -> Gripper Ready 最大单关节变化 = %.6f rad",
        ready_delta
    );

    if (ready_delta >
        JOINT_STEP_MAX * static_cast<double>(MAX_SEGMENTS))
    {
        RCLCPP_ERROR(
            logger,
            "回 Ready 需要超过 %zu 个 0.10 rad 逻辑段；保持夹紧并停止",
            MAX_SEGMENTS
        );

        return finish(13);
    }

    if (ready_delta > JOINT_GOAL_TOL)
    {
        MoveGroup::Plan ready_combined;

        if (!buildBlendedTrajectory(
                group,
                ready_start,
                ready_state,
                ready_goal,
                ready_combined,
                "RETURN GRIPPER READY",
                logger))
        {
            RCLCPP_ERROR(
                logger,
                "回 Ready 分段规划检查失败；保持夹紧并停止"
            );

            return finish(14);
        }

        std::array<double, N> ready_before_execute{};
        moveit::core::RobotStatePtr ready_before_state;

        if (!readCurrent(
                group,
                ready_before_execute,
                ready_before_state))
        {
            RCLCPP_ERROR(
                logger,
                "回 Ready 执行前无法重新读取真实六轴；保持夹紧并停止"
            );

            return finish(15);
        }

        const double ready_start_change =
            maxDiff(ready_start, ready_before_execute);

        RCLCPP_INFO(
            logger,
            "回 Ready 执行前真实起点变化 = %.9f rad",
            ready_start_change
        );

        if (ready_start_change > START_STATE_TOL)
        {
            RCLCPP_ERROR(
                logger,
                "回 Ready 规划期间机械臂发生移动：%.6f > %.6f rad；保持夹紧并停止",
                ready_start_change,
                START_STATE_TOL
            );

            return finish(15);
        }

        RCLCPP_WARN(
            logger,
            "砖块保持夹紧，开始一次连续 Execute 返回 Gripper Ready"
        );

        if (!group.execute(ready_combined))
        {
            RCLCPP_ERROR(
                logger,
                "回 Gripper Ready 执行失败；保持夹紧并停止"
            );

            return finish(16);
        }

        std::this_thread::sleep_for(300ms);
    }

    std::array<double, N> ready_actual{};
    moveit::core::RobotStatePtr ready_final_state;

    if (!readCurrent(
            group,
            ready_actual,
            ready_final_state))
    {
        RCLCPP_ERROR(
            logger,
            "回 Ready 后无法读取实际六轴；夹爪仍保持闭合"
        );

        return finish(17);
    }

    const double ready_final_error =
        maxDiff(ready_actual, ready_goal);

    RCLCPP_INFO(
        logger,
        "Gripper Ready 最终实际最大关节误差 = %.9f rad",
        ready_final_error
    );

    if (ready_final_error > FINAL_ACTUAL_TOL)
    {
        RCLCPP_ERROR(
            logger,
            "回 Ready 最终误差超过 %.3f rad；夹爪仍保持闭合",
            FINAL_ACTUAL_TOL
        );

        return finish(18);
    }


    RCLCPP_INFO(
        logger,
        "===================================="
    );

    RCLCPP_INFO(
        logger,
        "VISION GRASP 完成 ✅"
    );

    RCLCPP_INFO(
        logger,
        "===================================="
    );

    return finish(0);
}
