#pragma once
// Apply ESTUN protocol hints only after checking the original MoveIt trajectory.
// Lift and other phases retain their original arrays and bridge defaults.
template<class JointTrajectory>
inline void applyDescentSpeedRequest(JointTrajectory& trajectory, bool descending) {
 if (!descending) return;
 for (auto& point : trajectory.points) {
  point.velocities.assign(trajectory.joint_names.size(), .20);
  point.accelerations.assign(trajectory.joint_names.size(), .10);
 }
}
