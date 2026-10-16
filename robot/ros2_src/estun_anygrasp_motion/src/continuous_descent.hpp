#pragma once
// Cartesian chunks are planned and checked separately, then retain geometry samples
// and execute once. Protocol timestamps are placeholders for the ESTUN bridge.
inline moveit_msgs::msg::RobotTrajectory planContinuousDescent(MoveGroup& group,moveit::core::RobotState& state,planning_scene::PlanningScene& scene,const geometry_msgs::msg::Pose& target,const rclcpp::Node::SharedPtr& node) {
 const auto initial=poseOf(state);const Eigen::Vector3d from(initial.position.x,initial.position.y,initial.position.z),to(target.position.x,target.position.y,target.position.z);
 const Eigen::Quaterniond orientation(target.orientation.w,target.orientation.x,target.orientation.y,target.orientation.z);
 const auto initial_q=getJoints(state);const double length=(to-from).norm();
 if(!std::isfinite(length)||length>.151)throw std::runtime_error("Continuous descent above150mm");
 const int chunks=std::max(1,static_cast<int>(std::ceil(length/.020)));
 moveit_msgs::msg::RobotTrajectory out;auto& merged=out.joint_trajectory;merged.joint_names.assign(names.begin(),names.end());
 auto append=[&](const Joints&q){trajectory_msgs::msg::JointTrajectoryPoint p;p.positions.assign(q.begin(),q.end());p.time_from_start.sec=static_cast<int>(merged.points.size());merged.points.push_back(p);};append(initial_q);
 Joints retained=initial_q;Eigen::Vector3d retained_p=from;size_t raw_count=0;
 for(int k=1;k<=chunks;++k){
  const auto chunk_start=getJoints(state);group.setStartState(state);auto next=target;const auto destination=from+(to-from)*double(k)/chunks;next.position.x=destination.x();next.position.y=destination.y();next.position.z=destination.z();
  moveit_msgs::msg::RobotTrajectory part;moveit_msgs::msg::MoveItErrorCodes error;double fraction=group.computeCartesianPath(std::vector<geometry_msgs::msg::Pose>{next},.005,part,true,&error);
  if(!std::isfinite(fraction)||fraction<.999999||error.val!=moveit_msgs::msg::MoveItErrorCodes::SUCCESS||part.joint_trajectory.points.empty())throw std::runtime_error("Continuous descent chunk incomplete");
  auto indices=joint_indices(part.joint_trajectory.joint_names);Joints previous=chunk_start;
  const auto& points=part.joint_trajectory.points;raw_count+=points.size();
  for(size_t n=0;n<points.size();++n){const auto&p=points[n];
   if(p.positions.size()!=6||(!p.velocities.empty()&&p.velocities.size()!=6)||(!p.accelerations.empty()&&p.accelerations.size()!=6))throw std::runtime_error("Malformed descent arrays");
   Joints q{};for(size_t j=0;j<6;++j){q[j]=p.positions[indices[j]];
    if(!std::isfinite(q[j])||(!p.velocities.empty()&&(!std::isfinite(p.velocities[indices[j]])||std::abs(p.velocities[indices[j]])>.101))||(!p.accelerations.empty()&&(!std::isfinite(p.accelerations[indices[j]])||std::abs(p.accelerations[indices[j]])>.101)))throw std::runtime_error("Original descent trajectory nonfinite or speed/acc invalid");}
   if(difference(q,chunk_start)>.30||difference(q,previous)>.10+1e-9)throw std::runtime_error("Descent chunk joint change invalid");previous=q;setJoints(state,q);const auto fk=state.getGlobalLinkTransform("tool0");
   if(!state.satisfiesBounds(state.getJointModelGroup("arm"))||!scene.isStateValid(state,"arm"))throw std::runtime_error("Descent state invalid");
   const bool final_point=n+1==points.size();
   if(!final_point&&(fk.translation()-retained_p).norm()<.004&&difference(q,retained)<.02)continue;
   if(difference(q,retained)<1e-9)continue;
   if((fk.translation()-retained_p).norm()>.006||difference(q,retained)>.10)throw std::runtime_error("Retained descent geometry gap too large");
   // Check the actual retained joint chords against the full Cartesian line,
   // including scene validity, at <=1 mm and <=0.002 rad sampling intervals.
   int samples=std::max({1,static_cast<int>(std::ceil(difference(q,retained)/.002)),static_cast<int>(std::ceil((fk.translation()-retained_p).norm()/.001))});
   auto sample_state=state;
   for(int j=0;j<=samples;++j){Joints sample{};for(size_t a=0;a<6;++a)sample[a]=retained[a]+double(j)/samples*(q[a]-retained[a]);setJoints(sample_state,sample);const auto t=sample_state.getGlobalLinkTransform("tool0");
    double along=length>1e-9?(t.translation()-from).dot(to-from)/(length*length):0.;const auto nearest=from+std::clamp(along,0.,1.)*(to-from);
    if((t.translation()-nearest).norm()>.001||Eigen::Quaterniond(t.rotation()).angularDistance(orientation)>.5*M_PI/180.||!sample_state.satisfiesBounds(sample_state.getJointModelGroup("arm"))||!scene.isStateValid(sample_state,"arm"))throw std::runtime_error("Retained descent chord fails Cartesian/scene check");
   }
   append(q);retained=q;retained_p=fk.translation();
  }
  setJoints(state,previous);if((state.getGlobalLinkTransform("tool0").translation()-destination).norm()>.001)throw std::runtime_error("Descent chunk endpoint error above1mm");
 }
 applyDescentSpeedRequest(merged,true);
 RCLCPP_INFO(node->get_logger(),"CONTINUOUS DESCEND: %d checked chunks; %zu raw -> %zu geometry points; one Execute; velocity0.20 acceleration0.10",chunks,raw_count,merged.points.size());
 return out;
}
inline void executeContinuousDescent(MoveGroup& group,Feedback& feedback,RobotStatus& status,planning_scene::PlanningScene& scene,const rclcpp::Node::SharedPtr& node,const geometry_msgs::msg::Pose& target) {
 if(!status.ready())throw std::runtime_error("Robot not ready before continuous descent");const auto current=feedback.read(node);auto state=group.getCurrentState(5.);if(!state)throw std::runtime_error("Missing descent state");setJoints(*state,current);auto start_state=*state;
 MoveGroup::Plan plan;moveit::core::robotStateToRobotStateMsg(start_state,plan.start_state);plan.trajectory=planContinuousDescent(group,*state,scene,target,node);const auto goal=getJoints(*state);
 if(difference(current,feedback.read(node))>.03||!status.ready())throw std::runtime_error("Continuous descent start/status changed");
 if(group.execute(plan)!=moveit::core::MoveItErrorCode::SUCCESS)throw std::runtime_error("Continuous descent Execute failed; no retry");
 std::this_thread::sleep_for(100ms);verifyEndpoint(group,feedback,status,node,goal,&target);
}
