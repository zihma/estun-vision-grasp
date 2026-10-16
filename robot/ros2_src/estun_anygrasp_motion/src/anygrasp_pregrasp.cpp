#include "blended.hpp"
#include "robot_status.hpp"
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/robot_state/conversions.hpp>
#include <moveit_msgs/msg/display_trajectory.hpp>
#include <moveit_msgs/srv/get_planning_scene.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <codroid_msgs/msg/robot_info.hpp>
#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>
#ifdef ANYGRASP_FULL_CYCLE
#include <step_motor/msg/motor.hpp>
#endif
#include <chrono>
#include <fstream>
#include <mutex>
#include <thread>
#include <future>
#include <iostream>
#include <map>
using namespace std::chrono_literals;
using MoveGroup=moveit::planning_interface::MoveGroupInterface;
using namespace estun_anygrasp;
namespace {
struct SpinGuard {
 rclcpp::executors::MultiThreadedExecutor executor; std::thread thread;
 explicit SpinGuard(const rclcpp::Node::SharedPtr& node) {executor.add_node(node);thread=std::thread([this]{executor.spin();});}
 ~SpinGuard(){rclcpp::shutdown();executor.cancel();if(thread.joinable())thread.join();}
};
double unixNow(){return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();}
void checkCapture(double capture){double age=unixNow()-capture;if(!std::isfinite(age)||age<0||age>120.)throw std::runtime_error("Capture not fresh within 120 seconds; recapture, never edit timestamp");}
struct Feedback {
 std::mutex mutex; Joints values{}; bool received=false;
 rclcpp::Time stamp{0,0,RCL_ROS_TIME}; std::chrono::steady_clock::time_point receipt{};
 void update(const sensor_msgs::msg::JointState& m) {
  try {
   Joints next{};for(size_t i=0;i<6;++i){auto it=std::find(m.name.begin(),m.name.end(),names[i]);
    if(it==m.name.end()||std::count(m.name.begin(),m.name.end(),names[i])!=1) return;
    auto index=static_cast<size_t>(it-m.name.begin());if(index>=m.position.size()||!std::isfinite(m.position[index]))return;
    next[i]=m.position[index];}
   std::lock_guard<std::mutex> lock(mutex);values=next;stamp=rclcpp::Time(m.header.stamp);receipt=std::chrono::steady_clock::now();received=true;
  }catch(const std::exception&){}
 }
 Joints read(const rclcpp::Node::SharedPtr& node) {
  std::lock_guard<std::mutex> lock(mutex);
  double receipt_age=std::chrono::duration<double>(std::chrono::steady_clock::now()-receipt).count();
  double stamp_age=(node->now()-stamp).seconds();
  if(!received||receipt_age>1.||stamp.nanoseconds()==0||stamp_age>1.||stamp_age<-.1)
   throw std::runtime_error("joint_states missing, stale or future stamped");
  return values;
 }
};
void setJoints(moveit::core::RobotState&state,const Joints&q){for(size_t i=0;i<6;++i)state.setVariablePosition(names[i],q[i]);state.update();}
Joints getJoints(const moveit::core::RobotState&state){Joints q{};for(size_t i=0;i<6;++i)q[i]=state.getVariablePosition(names[i]);difference(q,q);return q;}
void checkSegment(const MoveGroup::Plan&plan,const Joints&start,const Joints&goal) {
 const auto&t=plan.trajectory.joint_trajectory;
 auto indices=joint_indices(t.joint_names);if(t.points.empty())throw std::runtime_error("empty segment plan");
 if(!plan.trajectory.multi_dof_joint_trajectory.points.empty())throw std::runtime_error("unexpected multi-DOF trajectory");
 double velocity=0,acceleration=0,last_time=-1.;
 for(const auto&p:t.points) {
  if(p.positions.size()!=6||(!p.velocities.empty()&&p.velocities.size()!=6)||(!p.accelerations.empty()&&p.accelerations.size()!=6))throw std::runtime_error("malformed trajectory arrays");
  auto time=p.time_from_start.sec+p.time_from_start.nanosec*1e-9;
  if(time<0||time<last_time)throw std::runtime_error("invalid trajectory time");
  last_time=time;
  for(size_t j=0;j<6;++j){if(!std::isfinite(p.positions[j]))throw std::runtime_error("nonfinite position");
   if(!p.velocities.empty()){if(!std::isfinite(p.velocities[j]))throw std::runtime_error("nonfinite velocity");velocity=std::max(velocity,std::abs(p.velocities[j]));}
   if(!p.accelerations.empty()){if(!std::isfinite(p.accelerations[j]))throw std::runtime_error("nonfinite acceleration");acceleration=std::max(acceleration,std::abs(p.accelerations[j]));}}
 }
 Joints first{},last{};for(size_t i=0;i<6;++i){first[i]=t.points.front().positions[indices[i]];last[i]=t.points.back().positions[indices[i]];}
 if(difference(first,start)>.005||difference(last,goal)>.005||difference(start,goal)>.10+1e-9||velocity>.095+1e-9||acceleration>.095+1e-9)
  throw std::runtime_error("segment start/end, step, velocity or acceleration gate failed");
}
#ifdef ANYGRASP_FULL_CYCLE
#include "cycle_helpers.hpp"
#include "continuous_descent.hpp"
#endif
struct Solution {Joints q;geometry_msgs::msg::Pose pose;int rank,symmetry;double score,delta;};
}
int main(int argc,char**argv) {
 rclcpp::init(argc,argv);
 auto node=rclcpp::Node::make_shared("anygrasp_pregrasp",rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));
 SpinGuard spin(node);
 auto param=[&](const std::string&key,auto value){if(!node->has_parameter(key))node->declare_parameter(key,value);return node->get_parameter(key).get_value<decltype(value)>();};
 try {
  const bool execute=param("execute",false);
#if defined(ANYGRASP_POSITIONING_ONLY) || defined(ANYGRASP_FULL_CYCLE)
  const bool operator_reviewed=param("operator_reviewed",false);
  if(execute&&!operator_reviewed)throw std::runtime_error("Physical path review must be acknowledged with operator_reviewed=true");
#else
  if(execute)throw std::runtime_error("This interface is PLAN-ONLY; execute=true is rejected");
#endif
  const auto mode=param("state_source",std::string("live"));
  if(mode!="live"&&mode!="offline")throw std::runtime_error("state_source must be live or offline");
  if(execute&&mode!="live")throw std::runtime_error("Execution requires live joint feedback; offline execution is rejected");
  auto input=param("candidate_file",std::string(""));auto output=param("report_file",std::string("/tmp/anygrasp_moveit_plan.yaml"));
  auto max_delta=param("max_joint_delta",2.);auto min_score=param("min_model_score",0.);
  if(!std::isfinite(max_delta)||max_delta<.1||max_delta>2.||!std::isfinite(min_score)||min_score<0||min_score>1)throw std::runtime_error("invalid joint/score gate");
#ifdef ANYGRASP_FULL_CYCLE
  const double min_clearance=param("min_tip_table_clearance_m",0.0);
  if(!std::isfinite(min_clearance)||min_clearance<0.0||min_clearance>.100)throw std::runtime_error("min_tip_table_clearance_m must be between 0.000 and 0.100 m");
  RCLCPP_INFO(node->get_logger(),"Physical fingertip table clearance gate: %.1f mm (point-cloud estimate)",min_clearance*1000.);
  size_t clearance_rejected=0;
#endif
  YAML::Node data=YAML::LoadFile(input);
#if defined(ANYGRASP_POSITIONING_ONLY) || defined(ANYGRASP_FULL_CYCLE)
#ifdef ANYGRASP_FULL_CYCLE
  if(data["mode"].as<std::string>()!="FULL_GRASP_CYCLE"||!data["robot_execution_allowed"].as<bool>())throw std::runtime_error("Full-cycle bundle required");
#else
  if(data["mode"].as<std::string>()!="PREGRASP_POSITIONING_ONLY"||!data["robot_execution_allowed"].as<bool>())throw std::runtime_error("Explicit positioning-only bundle required");
#endif
#else
  if(data["mode"].as<std::string>()!="PLAN_ONLY"||data["robot_execution_allowed"].as<bool>())throw std::runtime_error("Plan-only bundle required");
#endif
  if(data["schema_version"].as<int>()!=1||data["frame"].as<std::string>()!="base_link"||data["unit"].as<std::string>()!="m")throw std::runtime_error("invalid bundle schema/frame/unit");
  const double now_s=std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
  const double capture_s=data["capture_unix_s"].as<double>();
  if(!std::isfinite(capture_s))throw std::runtime_error("invalid capture timestamp");
  if(execute)checkCapture(capture_s);
  RCLCPP_WARN(node->get_logger(),"capture age %.1f s; execute=%s; score is not physical success probability",now_s-capture_s,execute?"true":"false");
  RobotStatus status;
  auto status_sub=node->create_subscription<codroid_msgs::msg::RobotInfo>("/RobotInfo",10,[&](codroid_msgs::msg::RobotInfo::SharedPtr m){status.update(m->state,m->fault_flag);});
  Feedback feedback;auto topic=mode=="offline"?"/anygrasp_offline/joint_states":"/joint_states";
  auto sub=node->create_subscription<sensor_msgs::msg::JointState>(topic,10,[&](sensor_msgs::msg::JointState::SharedPtr m){feedback.update(*m);});
#ifdef ANYGRASP_FULL_CYCLE
  auto motor=node->create_publisher<step_motor::msg::Motor>("/motor_control",10);
#endif
  MoveGroup group(node,"arm");group.setPoseReferenceFrame("base_link");group.setEndEffectorLink("tool0");
  group.setPlanningTime(5.);group.setNumPlanningAttempts(5);group.setMaxVelocityScalingFactor(.03);group.setMaxAccelerationScalingFactor(.03);
  for(int i=0;i<100;++i){try{feedback.read(node);break;}catch(const std::exception&){std::this_thread::sleep_for(50ms);}}
  if(execute) {
   if(!status.waitForFirst(5s))throw std::runtime_error("RobotInfo discovery timeout after 5 s: "+status.describe());
   if(!status.ready())throw std::runtime_error("RobotInfo not fresh AutoReady/fault=0 before planning: "+status.describe());
  }
  Joints current=feedback.read(node);auto state=group.getCurrentState(5.);
  if(!state)throw std::runtime_error("MoveIt current state unavailable");
  setJoints(*state,current);
#ifdef ANYGRASP_FULL_CYCLE
  const auto ready_values=data["offline_seed_joints"].as<std::vector<double>>();if(ready_values.size()!=6)throw std::runtime_error("Invalid Ready reference");Joints cycle_ready{};std::copy_n(ready_values.begin(),6,cycle_ready.begin());
  if(execute&&difference(current,cycle_ready)>.03)throw std::runtime_error("Return to Gripper Ready before capturing a full-cycle trial");
#endif
  const auto*jmg=state->getJointModelGroup("arm");if(!jmg||jmg->getVariableCount()!=6)throw std::runtime_error("arm must contain six joints");
  const auto vars=jmg->getVariableNames();joint_indices(vars);
  auto client=node->create_client<moveit_msgs::srv::GetPlanningScene>("/get_planning_scene");
  if(!client->wait_for_service(5s))throw std::runtime_error("planning scene service unavailable");
  auto request=std::make_shared<moveit_msgs::srv::GetPlanningScene::Request>();request->components.components=1023;
  auto future=client->async_send_request(request);if(future.wait_for(5s)!=std::future_status::ready)throw std::runtime_error("planning scene timeout");
  planning_scene::PlanningScene scene(group.getRobotModel());scene.setPlanningSceneMsg(future.get()->scene);
  if(!scene.isStateValid(*state,"arm"))throw std::runtime_error("initial state invalid in planning scene");
  MoveGroup::Plan combined;Solution selected;YAML::Node segments;bool planned=false;
  int reference_attempts=1;
#ifdef ANYGRASP_FULL_CYCLE
  double active_reference_m=data["geometry"]["flange_to_grasp_reference_m"].as<double>();
  bool reference_fallback_used=false;
  std::vector<moveit_msgs::msg::RobotTrajectory> cycle_preview;
  if(data["fallback_candidates"]) {
   if(std::abs(active_reference_m-.150)>1e-8||
      std::abs(data["fallback_geometry"]["flange_to_grasp_reference_m"].as<double>()-.160)>1e-8||
      std::abs(data["geometry"]["physical_flange_to_tip_m"].as<double>()-.180)>1e-8||
      std::abs(data["fallback_geometry"]["physical_flange_to_tip_m"].as<double>()-.180)>1e-8||
      data["fallback_candidates"].size()!=data["candidates"].size())throw std::runtime_error("Invalid 150 -> 160 mm fallback metadata");
   reference_attempts=2;
  }
#endif
  for(int reference_attempt=0;reference_attempt<reference_attempts&&!planned;++reference_attempt) {
#ifdef ANYGRASP_FULL_CYCLE
   if(reference_attempt==1) {
    if(execute){checkCapture(capture_s);if(!status.ready())throw std::runtime_error("Robot status invalid before 160 mm fallback");}
    if(difference(current,feedback.read(node))>.03)throw std::runtime_error("Start drift before 160 mm fallback");
    auto refresh=client->async_send_request(request);if(refresh.wait_for(5s)!=std::future_status::ready)throw std::runtime_error("Fallback scene refresh timeout");
    scene.setPlanningSceneMsg(refresh.get()->scene);
    if(!scene.isStateValid(*state,"arm"))throw std::runtime_error("Initial state invalid before fallback");
    data["candidates"]=YAML::Clone(data["fallback_candidates"]);
    data["geometry"]=YAML::Clone(data["fallback_geometry"]);
    active_reference_m=.160;reference_fallback_used=true;
    RCLCPP_WARN(node->get_logger(),"All 150 mm candidates failed preflight; trying 160 mm with the same capture. No motion has been requested.");
   }
   clearance_rejected=0;
   RCLCPP_INFO(node->get_logger(),"Candidate preflight grasp reference: %.0f mm",active_reference_m*1000.);
#endif
  std::vector<Solution> solutions;
  for(const auto&candidate:data["candidates"]) {
   int rank=candidate["rank"].as<int>();double score=candidate["score"].as<double>();double width=candidate["opening_width_m"].as<double>();
   double tilt=candidate["tilt_from_down_deg"].as<double>();
   if(!std::isfinite(score)||score<min_score||score>1||!std::isfinite(width)||width<=0||width+.010>.120+1e-7||!std::isfinite(tilt)||tilt<0||tilt>25.)continue;
   auto xyz=candidate["pregrasp_xyz_m"].as<std::vector<double>>();if(xyz.size()!=3||!std::all_of(xyz.begin(),xyz.end(),[](double x){return std::isfinite(x)&&std::abs(x)<=2.;}))throw std::runtime_error("bad pregrasp XYZ");
   if(execute&&(xyz[2]<.15||xyz[2]>.65))continue;
#ifdef ANYGRASP_FULL_CYCLE
   double clearance=candidate["estimated_tip_table_clearance_m"].as<double>();
   if(!std::isfinite(clearance)||clearance<min_clearance){
    ++clearance_rejected;
    RCLCPP_WARN(node->get_logger(),"rank=%d rejected: physical tip clearance %.2f mm < required %.2f mm",rank,clearance*1000.,min_clearance*1000.);
    continue;
   }
#endif
   int symmetry=0;
   for(const auto&orientation:candidate["tool0_quaternions_xyzw"]) {
    auto q=orientation.as<std::vector<double>>();if(q.size()!=4)throw std::runtime_error("bad quaternion length");
    double norm=0;for(double v:q){if(!std::isfinite(v))throw std::runtime_error("nonfinite quaternion");norm+=v*v;}if(std::abs(norm-1)>.0001)throw std::runtime_error("quaternion not unit");
    geometry_msgs::msg::Pose pose;pose.position.x=xyz[0];pose.position.y=xyz[1];pose.position.z=xyz[2];pose.orientation.x=q[0];pose.orientation.y=q[1];pose.orientation.z=q[2];pose.orientation.w=q[3];
#ifdef ANYGRASP_FULL_CYCLE
    if(tilt>15.){++symmetry;continue;}
    Eigen::Quaterniond vertical_orientation(q[3],q[0],q[1],q[2]);
    const double actual_tilt=std::acos(std::clamp(-vertical_orientation.toRotationMatrix().col(2).z(),-1.,1.))*180./M_PI;
    if(actual_tilt>15.+1e-7){++symmetry;continue;}
    if(data["reference_pregrasp_xyz_m"]){
    const auto reference_xyz=data["reference_pregrasp_xyz_m"].as<std::vector<double>>();
    const auto reference_q=data["reference_pregrasp_quaternion_xyzw"].as<std::vector<double>>();
    if(reference_xyz.size()!=3||reference_q.size()!=4)throw std::runtime_error("Missing verified high-pose reference");
    Eigen::Vector3d reference(reference_xyz[0],reference_xyz[1],reference_xyz[2]);
    Eigen::Quaterniond reference_orientation(reference_q[3],reference_q[0],reference_q[1],reference_q[2]);
    if((Eigen::Vector3d(xyz[0],xyz[1],xyz[2])-reference).norm()>.010||Eigen::Quaterniond(q[3],q[0],q[1],q[2]).angularDistance(reference_orientation)>5.*M_PI/180.){++symmetry;continue;}
    }
#endif
    for(int seed=0;seed<3;++seed) {
     moveit::core::RobotState trial=*state;if(seed)trial.setToRandomPositions(jmg);
     if(!trial.setFromIK(jmg,pose,"tool0",.08))continue;
     trial.update();
     if(!trial.satisfiesBounds(jmg)||!scene.isStateValid(trial,"arm"))continue;
     const auto fk=trial.getGlobalLinkTransform("tool0");
     const Eigen::Vector3d desired(xyz[0],xyz[1],xyz[2]);
     const Eigen::Quaterniond orientation_target(q[3],q[0],q[1],q[2]);
     if((fk.translation()-desired).norm()>.01||Eigen::Quaterniond(fk.rotation()).angularDistance(orientation_target)>2.*M_PI/180.)continue;
     auto joints=getJoints(trial);double delta=difference(current,joints);if(delta>max_delta)continue;
     bool duplicate=false;for(const auto&s:solutions)if(s.rank==rank&&difference(s.q,joints)<1e-4)duplicate=true;
     if(!duplicate)solutions.push_back({joints,pose,rank,symmetry,score,delta});
    }
    ++symmetry;
   }
  }
  std::sort(solutions.begin(),solutions.end(),[](const auto&a,const auto&b){if(std::abs(a.delta-b.delta)>1e-6)return a.delta<b.delta;return a.score>b.score;});
#ifdef ANYGRASP_FULL_CYCLE
  RCLCPP_INFO(node->get_logger(),"Candidate clearance rejects: %zu/%zu; valid IK solutions: %zu",clearance_rejected,data["candidates"].size(),solutions.size());
  if(clearance_rejected==data["candidates"].size())RCLCPP_WARN(node->get_logger(),"All candidates rejected by physical fingertip table clearance; IK not attempted at this reference");
#endif

  for(const auto&solution:solutions) {
   try {
    auto points=waypoints(current,solution.q,max_delta);if(points.empty())throw std::runtime_error("already at goal; no motion needed");
    MoveGroup::Plan proposed;moveit::core::robotStateToRobotStateMsg(*state,proposed.start_state);
    proposed.trajectory.joint_trajectory.joint_names.assign(names.begin(),names.end());
    auto add=[&](const Joints&q,int sec){trajectory_msgs::msg::JointTrajectoryPoint p;p.positions.assign(q.begin(),q.end());p.velocities.assign(6,0.);p.accelerations.assign(6,0.);p.time_from_start.sec=sec;proposed.trajectory.joint_trajectory.points.push_back(p);};
    add(current,0);Joints previous=current;YAML::Node checked;
    for(size_t i=0;i<points.size();++i) {
     auto target=points[i];moveit::core::RobotState start=*state;setJoints(start,previous);group.setStartState(start);
     std::map<std::string,double> named;for(size_t j=0;j<6;++j)named[names[j]]=target[j];
     group.clearPoseTargets();if(!group.setJointValueTarget(named))throw std::runtime_error("joint target rejected");
     MoveGroup::Plan part;if(group.plan(part)!=moveit::core::MoveItErrorCode::SUCCESS)throw std::runtime_error("segment planning failed");checkSegment(part,previous,target);
     // Guide section 6.2: check the direct keypoint connection too, at <=0.01 rad.
     int samples=std::max(1,static_cast<int>(std::ceil(difference(previous,target)/.01)));
     for(int k=0;k<=samples;++k){Joints q{};for(size_t j=0;j<6;++j)q[j]=previous[j]+double(k)/samples*(target[j]-previous[j]);moveit::core::RobotState sampled=*state;setJoints(sampled,q);if(!sampled.satisfiesBounds(jmg)||!scene.isStateValid(sampled,"arm"))throw std::runtime_error("direct keypoint interpolation invalid");}
     YAML::Node entry;entry["segment"]=i+1;entry["delta_rad"]=difference(previous,target);entry["moveit_points_discarded"]=part.trajectory.joint_trajectory.points.size();entry["direct_samples_checked"]=samples+1;checked.push_back(entry);
     RCLCPP_INFO(node->get_logger(),"rank=%d W%02zu delta=%.6f dense=%zu (discarded), direct samples=%d",solution.rank,i+1,difference(previous,target),part.trajectory.joint_trajectory.points.size(),samples+1);
     add(target,static_cast<int>(i+1));previous=target;
    }
#ifdef ANYGRASP_FULL_CYCLE
    YAML::Node cycle_data;
    for(const auto&candidate:data["candidates"])if(candidate["rank"].as<int>()==solution.rank){cycle_data=YAML::Clone(candidate);break;}
    auto cycle_xyz=cycle_data["grasp_tool0_xyz_m"].as<std::vector<double>>();
    if(cycle_xyz.size()!=3||!std::all_of(cycle_xyz.begin(),cycle_xyz.end(),[](double x){return std::isfinite(x)&&std::abs(x)<=2.;}))throw std::runtime_error("Invalid full-cycle target");
    auto cycle_grasp=solution.pose;cycle_grasp.position.x=cycle_xyz[0];cycle_grasp.position.y=cycle_xyz[1];cycle_grasp.position.z=cycle_xyz[2];
    moveit::core::RobotState hypothetical=*state;setJoints(hypothetical,solution.q);
    std::vector<moveit_msgs::msg::RobotTrajectory> preview;
    preview.push_back(planContinuousDescent(group,hypothetical,scene,cycle_grasp,node));
    hypothetical=previewLinear(group,hypothetical,solution.pose,&preview);
    previewReturn(group,hypothetical,scene,cycle_ready,&preview);
    cycle_preview=std::move(preview);
#endif
    combined=proposed;selected=solution;segments=checked;planned=true;break;
   }catch(const std::exception&e){RCLCPP_WARN(node->get_logger(),"candidate rank=%d symmetry=%d rejected: %s",solution.rank,solution.symmetry,e.what());}
  }
  } // Exhaust every candidate at 150 mm before considering 160 mm.
  if(!planned)throw std::runtime_error("All candidates failed clearance/IK/blended/full-cycle preflight at every configured grasp reference");
#ifdef ANYGRASP_FULL_CYCLE
  // Apply protocol hints only to the checked, compact full-cycle trajectory.
  applyDescentSpeedRequest(combined.trajectory.joint_trajectory,true);
  RCLCPP_INFO(node->get_logger(),"FULL CYCLE ESTUN speed requests: pregrasp/down/lift/Ready=0.20; acceleration=0.10");
#endif
#ifdef ANYGRASP_FULL_CYCLE
  RCLCPP_INFO(node->get_logger(),"FULL-CYCLE PREFLIGHT: Cartesian down/lift and blended return all planned at %.0f mm; no motion yet",active_reference_m*1000.);
#endif
  // A plan-only snapshot must still be invalidated when its starting state changed.
  if(difference(current,feedback.read(node))>.03)throw std::runtime_error("start drift above 0.03 rad; plan invalidated");
  auto pub=node->create_publisher<moveit_msgs::msg::DisplayTrajectory>("/display_planned_path",rclcpp::QoS(1).transient_local());
  moveit_msgs::msg::DisplayTrajectory display;display.trajectory_start=combined.start_state;display.trajectory.push_back(combined.trajectory);
#ifdef ANYGRASP_FULL_CYCLE
  for(const auto&path:cycle_preview)display.trajectory.push_back(path);
#endif
  pub->publish(display);
  YAML::Node report;report["generated_unix_s"]=std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
  report["world_collision_objects"]=scene.getWorld()->getObjectIds();
  report["four_finger_collision_verified"]=false;
  report["mode"]="PLAN_ONLY";report["robot_execution"]=false;report["state_source"]=mode;report["selected_rank"]=selected.rank;report["symmetry"]=selected.symmetry;report["model_score"]=selected.score;report["max_joint_delta_rad"]=selected.delta;
  report["pregrasp_xyz_m"]=std::vector<double>{selected.pose.position.x,selected.pose.position.y,selected.pose.position.z};
  report["pregrasp_quaternion_xyzw"]=std::vector<double>{selected.pose.orientation.x,selected.pose.orientation.y,selected.pose.orientation.z,selected.pose.orientation.w};
#ifdef ANYGRASP_FULL_CYCLE
  report["max_approach_tilt_deg"]=15.;
  report["flange_to_grasp_reference_m"]=active_reference_m;
  report["reference_fallback_used"]=reference_fallback_used;
  report["physical_flange_to_tip_m"]=.180;
  report["min_tip_table_clearance_m"]=min_clearance;
  report["cycle_estun_velocity_request"]=.20;
  report["cycle_estun_acceleration_request"]=.10;
  report["descent_execution_mode"]="CONTINUOUS_CARTESIAN_ONE_EXECUTE";
  report["descent_estun_velocity_request"]=.20;
  report["descent_estun_acceleration_request"]=.10;
  report["clearance_rejected_candidates"]=clearance_rejected;
#endif
  report["candidate_file"]=input;report["source_capture_unix_s"]=data["capture_unix_s"];report["start_joints"]=std::vector<double>(current.begin(),current.end());report["goal_joints"]=std::vector<double>(selected.q.begin(),selected.q.end());report["segments"]=segments;
  report["joint_names"]=std::vector<std::string>(names.begin(),names.end());
  for(const auto&p:combined.trajectory.joint_trajectory.points){YAML::Node entry;entry["positions"]=p.positions;entry["time_index_s"]=p.time_from_start.sec;report["combined_keypoints"].push_back(entry);}
  report["limitations"].push_back("Modeled PlanningScene only; real four-finger collision geometry and physical contact offset are unverified");
  report["limitations"].push_back("Keypoint time/zero derivatives are ESTUN protocol placeholders, not a hardware speed guarantee");
  auto saveReport=[&](){std::ofstream file(output);if(!file)throw std::runtime_error("cannot write report");file<<report;file.close();if(!file)throw std::runtime_error("report write failed");};
  saveReport();
  RCLCPP_INFO(node->get_logger(),"PLAN-ONLY SUCCESS rank=%d symmetry=%d segments=%zu keypoints=%zu state=%s report=%s",selected.rank,selected.symmetry,segments.size(),combined.trajectory.joint_trajectory.points.size(),mode.c_str(),output.c_str());
#if defined(ANYGRASP_POSITIONING_ONLY) || defined(ANYGRASP_FULL_CYCLE)
  if(execute) {
#ifdef ANYGRASP_FULL_CYCLE
   if(motor->get_subscription_count()==0)throw std::runtime_error("No motor_control subscriber; full cycle stopped before movement");
   std::cout << "\nGrasp reference: " << active_reference_m*1000. << " mm; physical fingertip: 180 mm.\n";
   std::cout << "\nFULL GRASP CYCLE: open -> Cartesian down -> close -> lift -> Ready (remain closed).\n";
   const std::string required_word="GRASP";
#else
   const std::string required_word="POSITION";
#endif
   std::cout << "\nPREGRASP TARGET: rank " << selected.rank << " symmetry " << selected.symmetry
     << " score " << selected.score << "\ntool0 XYZ [" << selected.pose.position.x << " " << selected.pose.position.y << " " << selected.pose.position.z
     << "] m; " << segments.size() << " segments, " << combined.trajectory.joint_trajectory.points.size()
     << " keypoints. See phase logs for ESTUN speed requests (bridge maximum 0.20).\n"
     << "Inspect RViz and the actual full path. Type " << required_word << " to proceed: " << std::flush;
   std::string answer; if(!(std::cin>>answer)||answer!=required_word) {
    report["execution_status"]="CANCELLED";saveReport();RCLCPP_WARN(node->get_logger(),"Confirmation cancelled; no motion");return 0;
   }
   checkCapture(capture_s);
   if(!status.ready())throw std::runtime_error("RobotInfo invalid after confirmation: "+status.describe());
   auto before=feedback.read(node);if(difference(current,before)>.03)throw std::runtime_error("Start drift >0.03 rad; old plan rejected");
   // Refresh the scene and recheck the exact keypoint connections after confirmation.
   auto latest=client->async_send_request(request);if(latest.wait_for(5s)!=std::future_status::ready)throw std::runtime_error("Scene refresh timeout");
   scene.setPlanningSceneMsg(latest.get()->scene);
   const auto&keypoints=combined.trajectory.joint_trajectory.points;
   for(size_t i=1;i<keypoints.size();++i){Joints a{},b{};std::copy_n(keypoints[i-1].positions.begin(),6,a.begin());std::copy_n(keypoints[i].positions.begin(),6,b.begin());
    int count=std::max(1,static_cast<int>(std::ceil(difference(a,b)/.01)));
    for(int k=0;k<=count;++k){Joints q{};for(size_t j=0;j<6;++j)q[j]=a[j]+double(k)/count*(b[j]-a[j]);moveit::core::RobotState checked=*state;setJoints(checked,q);if(!checked.satisfiesBounds(jmg)||!scene.isStateValid(checked,"arm"))throw std::runtime_error("Confirmed direct path no longer valid");}}
   checkCapture(capture_s);
   if(!status.ready()||difference(current,feedback.read(node))>.03)throw std::runtime_error("Robot status/start changed before Execute");
#ifdef ANYGRASP_FULL_CYCLE
   report["mode"]="FULL_GRASP_CYCLE";
#else
   report["mode"]="PREGRASP_POSITIONING_ONLY";
#endif
   report["execution_attempted"]=true;report["robot_execution"]="REQUESTED_NOT_VERIFIED";report["execution_status"]="STARTED";saveReport();
   RCLCPP_WARN(node->get_logger(),"One blended Execute only: %zu keypoints; no automatic retry",keypoints.size());
   const auto result=group.execute(combined);
   if(result!=moveit::core::MoveItErrorCode::SUCCESS){report["robot_execution"]="NOT_VERIFIED";report["execution_status"]="FAILED_OR_ABORTED";saveReport();throw std::runtime_error("Positioning Execute failed/aborted; inspect controller and bridge, do not resend old plan");}
   bool ended=false;
   for(int i=0;i<50;++i){if(status.ready()){ended=true;break;}std::this_thread::sleep_for(100ms);}
   if(!ended){report["execution_status"]="FINAL_STATUS_FAILED";saveReport();throw std::runtime_error("No fresh AutoReady/fault=0 after Execute");}
   auto actual=feedback.read(node);double joint_error=difference(actual,selected.q);
   auto final_state=group.getCurrentState(5.);if(!final_state)throw std::runtime_error("No final state");setJoints(*final_state,actual);
   const auto fk=final_state->getGlobalLinkTransform("tool0");
   Eigen::Vector3d desired(selected.pose.position.x,selected.pose.position.y,selected.pose.position.z);
   Eigen::Quaterniond orientation(selected.pose.orientation.w,selected.pose.orientation.x,selected.pose.orientation.y,selected.pose.orientation.z);
   double position_error=(fk.translation()-desired).norm();double angle_error=Eigen::Quaterniond(fk.rotation()).angularDistance(orientation);
   report["actual_joints"]=std::vector<double>(actual.begin(),actual.end());report["joint_error_rad"]=joint_error;report["tool0_position_error_m"]=position_error;report["tool0_orientation_error_deg"]=angle_error*180./M_PI;
   report["execution_status"]=(joint_error<=.03&&position_error<=.01&&angle_error<=2.*M_PI/180.)?"VERIFIED_AT_PREGRASP":"ENDPOINT_CHECK_FAILED";
   report["robot_execution"]=true;saveReport();
   if(joint_error>.03||position_error>.01||angle_error>2.*M_PI/180.)throw std::runtime_error("Actual endpoint check failed; stop");
   RCLCPP_INFO(node->get_logger(),"PREGRASP VERIFIED: joint error %.6f rad, tool0 %.2f mm / %.2f deg; no descent or gripper command",joint_error,position_error*1000.,angle_error*180./M_PI);
#ifdef ANYGRASP_FULL_CYCLE
   bool close_issued=false;
   try {
    if(std::acos(std::clamp(-fk.rotation().col(2).z(),-1.,1.))*180./M_PI>15.)throw std::runtime_error("Actual tool tilt exceeds 15 degrees; do not descend");
    YAML::Node selected_data;bool found=false;
    for(size_t index=0;index<data["candidates"].size();++index)if(data["candidates"][index]["rank"].as<int>()==selected.rank){selected_data.reset(data["candidates"][index]);found=true;break;}
    if(!found)throw std::runtime_error("Selected candidate metadata missing");
    auto grasp_xyz=selected_data["grasp_tool0_xyz_m"].as<std::vector<double>>();if(grasp_xyz.size()!=3)throw std::runtime_error("Bad grasp XYZ");
    auto grasp_pose=selected.pose;grasp_pose.position.x=grasp_xyz[0];grasp_pose.position.y=grasp_xyz[1];grasp_pose.position.z=grasp_xyz[2];
    for(double x:grasp_xyz)if(!std::isfinite(x))throw std::runtime_error("Nonfinite grasp target");
    checkCapture(capture_s);
    auto sendMotor=[&](bool close){if(!status.ready()||motor->get_subscription_count()==0)throw std::runtime_error("Robot/motor unavailable before gripper command");step_motor::msg::Motor m;m.id=1;m.speed=200;m.dir=close?1:0;m.mode=2;m.angle=30000;m.state=0;m.sub_divide=32;motor->publish(m);std::this_thread::sleep_for(2s);};
    report["execution_status"]="OPENING";saveReport();sendMotor(false);checkCapture(capture_s);
    report["execution_status"]="DESCENDING";saveReport();
    auto descent_world=client->async_send_request(request);if(descent_world.wait_for(5s)!=std::future_status::ready)throw std::runtime_error("Descent scene refresh failed");scene.setPlanningSceneMsg(descent_world.get()->scene);
    executeContinuousDescent(group,feedback,status,scene,node,grasp_pose);
    report["execution_status"]="CLOSING";saveReport();close_issued=true;sendMotor(true);
    report["execution_status"]="LIFTING";saveReport();moveLinear(group,feedback,status,node,selected.pose,"LIFT");
    auto ready_vector=data["offline_seed_joints"].as<std::vector<double>>();if(ready_vector.size()!=6)throw std::runtime_error("Bad Ready reference");Joints ready{};std::copy_n(ready_vector.begin(),6,ready.begin());
    auto world=client->async_send_request(request);if(world.wait_for(5s)!=std::future_status::ready)throw std::runtime_error("Return scene refresh failed");scene.setPlanningSceneMsg(world.get()->scene);
    report["execution_status"]="RETURNING_READY";saveReport();returnReady(group,feedback,status,scene,node,ready);
    auto final_ready=feedback.read(node);report["final_ready_actual_joints"]=std::vector<double>(final_ready.begin(),final_ready.end());report["final_ready_joint_error_rad"]=difference(final_ready,ready);
    report["execution_status"]="CYCLE_MOTION_COMPLETE_GRIPPER_CLOSED";report["gripper_close_command_sent"]=true;report["physical_object_holding_verified"]=false;saveReport();
    RCLCPP_INFO(node->get_logger(),"FULL CYCLE MOTION COMPLETE: gripper remains closed; visually confirm tube held and intact");
   } catch(const std::exception&error) {
    report["execution_status"]="CYCLE_STOPPED";report["failure_reason"]=error.what();report["gripper_close_command_sent"]=close_issued;saveReport();
    if(close_issued)RCLCPP_ERROR(node->get_logger(),"Keep gripper closed; operator must handle held object. No automatic release or retry");
    throw;
   }
#endif
  }
#endif
  std::this_thread::sleep_for(300ms);return 0;
 }catch(const std::exception&e){RCLCPP_ERROR(node->get_logger(),"PLAN-ONLY STOP: %s",e.what());return 1;}
}
