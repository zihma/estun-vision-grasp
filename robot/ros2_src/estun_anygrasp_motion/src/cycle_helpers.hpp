#pragma once
#include "descent_speed_request.hpp"

inline geometry_msgs::msg::Pose poseOf(const moveit::core::RobotState& state) {
 const auto t=state.getGlobalLinkTransform("tool0");const Eigen::Quaterniond q(t.rotation());geometry_msgs::msg::Pose p;
 p.position.x=t.translation().x();p.position.y=t.translation().y();p.position.z=t.translation().z();p.orientation.x=q.x();p.orientation.y=q.y();p.orientation.z=q.z();p.orientation.w=q.w();return p;
}
inline void verifyEndpoint(MoveGroup& group,Feedback& feedback,RobotStatus& status,const rclcpp::Node::SharedPtr& node,const Joints& goal,const geometry_msgs::msg::Pose* pose=nullptr) {
 if(!status.ready())throw std::runtime_error("Endpoint robot status invalid: "+status.describe());
 auto q=feedback.read(node);if(difference(q,goal)>.03)throw std::runtime_error("Actual joint endpoint error >0.03 rad");
 if(pose){auto state=group.getCurrentState(5.);if(!state)throw std::runtime_error("No endpoint RobotState");setJoints(*state,q);auto t=state->getGlobalLinkTransform("tool0");Eigen::Vector3d p(pose->position.x,pose->position.y,pose->position.z);Eigen::Quaterniond r(pose->orientation.w,pose->orientation.x,pose->orientation.y,pose->orientation.z);
  if(std::acos(std::clamp(-t.rotation().col(2).z(),-1.,1.))*180./M_PI>15.|| (t.translation()-p).norm()>.01||Eigen::Quaterniond(t.rotation()).angularDistance(r)>2.*M_PI/180.)throw std::runtime_error("Actual TCP endpoint error >10 mm or 2 deg");}
}
inline void moveLinear(MoveGroup& group,Feedback& feedback,RobotStatus& status,const rclcpp::Node::SharedPtr& node,const geometry_msgs::msg::Pose& target,const std::string& label) {
 auto state=group.getCurrentState(5.);if(!state)throw std::runtime_error("No Cartesian start state");setJoints(*state,feedback.read(node));auto initial=poseOf(*state);
 Eigen::Vector3d start(initial.position.x,initial.position.y,initial.position.z),end(target.position.x,target.position.y,target.position.z);
 const double distance=(end-start).norm();if(distance>.151)throw std::runtime_error("Cartesian movement above 150 mm");const int segments=std::max(1,static_cast<int>(std::ceil(distance/.020)));
 for(int i=1;i<=segments;++i){
  if(!status.ready())throw std::runtime_error("Cartesian robot not ready: "+status.describe());
  auto current=feedback.read(node);state=group.getCurrentState(5.);if(!state)throw std::runtime_error("No fresh Cartesian state");setJoints(*state,current);group.setStartState(*state);
  auto next=target;auto p=start+(end-start)*double(i)/segments;next.position.x=p.x();next.position.y=p.y();next.position.z=p.z();
  moveit_msgs::msg::RobotTrajectory trajectory;moveit_msgs::msg::MoveItErrorCodes error;
  double fraction=group.computeCartesianPath(std::vector<geometry_msgs::msg::Pose>{next},.005,trajectory,true,&error);
  if(!std::isfinite(fraction)||fraction<.999999||error.val!=moveit_msgs::msg::MoveItErrorCodes::SUCCESS)throw std::runtime_error(label+": incomplete Cartesian path");
  const auto&jt=trajectory.joint_trajectory;auto indices=joint_indices(jt.joint_names);if(jt.points.empty())throw std::runtime_error("Empty Cartesian trajectory");
  Joints previous=current,goal{};
  for(const auto&point:jt.points){if(point.positions.size()!=6||(!point.velocities.empty()&&point.velocities.size()!=6)||(!point.accelerations.empty()&&point.accelerations.size()!=6))throw std::runtime_error("Malformed Cartesian arrays");
   Joints q{};for(size_t j=0;j<6;++j){q[j]=point.positions[indices[j]];
    if(!point.velocities.empty()&&(!std::isfinite(point.velocities[indices[j]])||std::abs(point.velocities[indices[j]])>.101))throw std::runtime_error("Cartesian speed invalid");
    if(!point.accelerations.empty()&&(!std::isfinite(point.accelerations[indices[j]])||std::abs(point.accelerations[indices[j]])>.101))throw std::runtime_error("Cartesian acceleration invalid");}
   if(difference(q,current)>.30||difference(q,previous)>.10+1e-9)throw std::runtime_error("Cartesian joint change invalid");
   previous=q;goal=q;
  }
  if(difference(current,feedback.read(node))>.03||!status.ready())throw std::runtime_error("Cartesian start/status changed");
  // Same per-trajectory ESTUN request mechanism as foam_brick_grasp.cpp.
  // Keep MoveIt planning checks above separate from protocol speed hints.
  applyDescentSpeedRequest(trajectory.joint_trajectory,true);
  RCLCPP_INFO(node->get_logger(),"%s ESTUN velocity request=%.2f acceleration=0.10",label.c_str(),.20);
  // Preserve Cartesian geometry points, unlike joint point-to-point simplification.
  MoveGroup::Plan plan;moveit::core::robotStateToRobotStateMsg(*state,plan.start_state);plan.trajectory=trajectory;
  RCLCPP_INFO(node->get_logger(),"%s %d/%d: fraction %.6f points %zu",label.c_str(),i,segments,fraction,jt.points.size());
  if(group.execute(plan)!=moveit::core::MoveItErrorCode::SUCCESS)throw std::runtime_error(label+": Execute failed; no retry");
  std::this_thread::sleep_for(100ms);verifyEndpoint(group,feedback,status,node,goal,&next);
 }
}
inline void returnReady(MoveGroup& group,Feedback& feedback,RobotStatus& status,planning_scene::PlanningScene& scene,const rclcpp::Node::SharedPtr& node,const Joints& goal) {
 auto current=feedback.read(node);auto points=waypoints(current,goal,2.);if(points.empty())return;
 auto state=group.getCurrentState(5.);if(!state)throw std::runtime_error("No return start state");setJoints(*state,current);
 MoveGroup::Plan combined;moveit::core::robotStateToRobotStateMsg(*state,combined.start_state);auto&jt=combined.trajectory.joint_trajectory;jt.joint_names.assign(names.begin(),names.end());
 auto add=[&](const Joints&q,int i){trajectory_msgs::msg::JointTrajectoryPoint p;p.positions.assign(q.begin(),q.end());p.velocities.assign(6,0.);p.accelerations.assign(6,0.);p.time_from_start.sec=i;jt.points.push_back(p);};add(current,0);Joints previous=current;
 for(size_t i=0;i<points.size();++i){auto q=points[i];setJoints(*state,previous);group.setStartState(*state);std::map<std::string,double> named;for(size_t j=0;j<6;++j)named[names[j]]=q[j];group.clearPoseTargets();if(!group.setJointValueTarget(named))throw std::runtime_error("Return target rejected");MoveGroup::Plan part;
  if(group.plan(part)!=moveit::core::MoveItErrorCode::SUCCESS)throw std::runtime_error("Return segment plan failed");
  checkSegment(part,previous,q);
  int count=std::max(1,static_cast<int>(std::ceil(difference(previous,q)/.01)));for(int k=0;k<=count;++k){Joints sample{};for(size_t j=0;j<6;++j)sample[j]=previous[j]+double(k)/count*(q[j]-previous[j]);setJoints(*state,sample);if(!scene.isStateValid(*state,"arm"))throw std::runtime_error("Return direct path invalid");}
  add(q,static_cast<int>(i+1));previous=q;
 }
 if(difference(current,feedback.read(node))>.03||!status.ready())throw std::runtime_error("Return start/status changed");
 applyDescentSpeedRequest(combined.trajectory.joint_trajectory,true);
 RCLCPP_INFO(node->get_logger(),"RETURN READY ESTUN velocity request=0.20 acceleration=0.10");
 if(group.execute(combined)!=moveit::core::MoveItErrorCode::SUCCESS)throw std::runtime_error("Return Execute failed; keep gripper closed");
 std::this_thread::sleep_for(100ms);verifyEndpoint(group,feedback,status,node,goal);
}
inline moveit::core::RobotState previewLinear(MoveGroup& group,moveit::core::RobotState state,const geometry_msgs::msg::Pose& target,std::vector<moveit_msgs::msg::RobotTrajectory>* shown=nullptr) {
 auto pose=poseOf(state);Eigen::Vector3d from(pose.position.x,pose.position.y,pose.position.z),to(target.position.x,target.position.y,target.position.z);
 int n=std::max(1,static_cast<int>(std::ceil((to-from).norm()/.020)));
 for(int k=1;k<=n;++k){auto next=target;auto p=from+(to-from)*double(k)/n;next.position.x=p.x();next.position.y=p.y();next.position.z=p.z();group.setStartState(state);
  moveit_msgs::msg::RobotTrajectory trajectory;moveit_msgs::msg::MoveItErrorCodes error;double fraction=group.computeCartesianPath(std::vector<geometry_msgs::msg::Pose>{next},.005,trajectory,true,&error);
  if(!std::isfinite(fraction)||fraction<.999999||error.val!=moveit_msgs::msg::MoveItErrorCodes::SUCCESS||trajectory.joint_trajectory.points.empty())throw std::runtime_error("Full-cycle Cartesian preflight failed");
  auto indices=joint_indices(trajectory.joint_trajectory.joint_names);auto start=getJoints(state);Joints previous=start;
  for(const auto&point:trajectory.joint_trajectory.points){if(point.positions.size()!=6)throw std::runtime_error("Bad preflight positions");Joints q{};for(size_t j=0;j<6;++j)q[j]=point.positions[indices[j]];
   if(difference(start,q)>.30||difference(previous,q)>.10+1e-9)throw std::runtime_error("Cartesian preflight joint change failed");
   previous=q;
  }
  if(shown)shown->push_back(trajectory);
  setJoints(state,previous);
 }
 return state;
}
inline void previewReturn(MoveGroup& group,moveit::core::RobotState state,planning_scene::PlanningScene& scene,const Joints& goal,std::vector<moveit_msgs::msg::RobotTrajectory>* shown=nullptr) {
 moveit_msgs::msg::RobotTrajectory compact;compact.joint_trajectory.joint_names.assign(names.begin(),names.end());int seconds=0;
 auto add=[&](const Joints&q){trajectory_msgs::msg::JointTrajectoryPoint p;p.positions.assign(q.begin(),q.end());p.time_from_start.sec=seconds++;compact.joint_trajectory.points.push_back(p);};
 auto previous=getJoints(state);add(previous);for(const auto&q:waypoints(previous,goal,2.)){
  setJoints(state,previous);group.setStartState(state);std::map<std::string,double> named;for(size_t j=0;j<6;++j)named[names[j]]=q[j];group.clearPoseTargets();if(!group.setJointValueTarget(named))throw std::runtime_error("Return preflight target failed");MoveGroup::Plan part;
  if(group.plan(part)!=moveit::core::MoveItErrorCode::SUCCESS)throw std::runtime_error("Return preflight plan failed");
  checkSegment(part,previous,q);
  int count=std::max(1,static_cast<int>(std::ceil(difference(previous,q)/.01)));for(int k=0;k<=count;++k){Joints sample{};for(size_t j=0;j<6;++j)sample[j]=previous[j]+double(k)/count*(q[j]-previous[j]);setJoints(state,sample);if(!scene.isStateValid(state,"arm"))throw std::runtime_error("Return preflight direct path failed");}
  add(q);previous=q;
 }
 if(shown&&compact.joint_trajectory.points.size()>1)shown->push_back(compact);
}
