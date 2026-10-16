#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
namespace estun_anygrasp {
using Joints = std::array<double,6>;
inline const std::array<std::string,6> names={"joint1","joint2","joint3","joint4","joint5","joint6"};
inline constexpr double step_max=.10, endpoint_tolerance=.005;
inline constexpr std::size_t max_segments=20;
inline double difference(const Joints&a,const Joints&b) {
 double d=0; for(size_t i=0;i<6;++i) {
  if(!std::isfinite(a[i])||!std::isfinite(b[i])) throw std::runtime_error("nonfinite joint");
  d=std::max(d,std::abs(a[i]-b[i]));
 } return d;
}
inline std::vector<Joints> waypoints(const Joints&start,const Joints&goal,double max_delta) {
 if(!std::isfinite(max_delta)||max_delta<.10||max_delta>2.) throw std::runtime_error("max_delta outside [0.10,2.00]");
 const auto total=difference(start,goal);
 if(total>max_delta+1e-9) throw std::runtime_error("total joint change exceeds configured limit");
 std::vector<Joints> out; Joints previous=start;
 while(difference(previous,goal)>1e-9) {
  if(out.size()>=max_segments) throw std::runtime_error("more than 20 segments");
  double remaining=difference(previous,goal); Joints next=goal;
  if(remaining>step_max) for(size_t i=0;i<6;++i) next[i]=previous[i]+step_max/remaining*(goal[i]-previous[i]);
  if(difference(previous,next)>step_max+1e-9) throw std::runtime_error("segment above 0.10 rad");
  out.push_back(next); previous=next;
 } return out;
}
inline std::array<size_t,6> joint_indices(const std::vector<std::string>& trajectory_names) {
 if(trajectory_names.size()!=6) throw std::runtime_error("expected six trajectory joints");
 std::array<size_t,6> result{};
 for(size_t i=0;i<6;++i) {
  auto it=std::find(trajectory_names.begin(),trajectory_names.end(),names[i]);
  if(it==trajectory_names.end()||std::count(trajectory_names.begin(),trajectory_names.end(),names[i])!=1)
   throw std::runtime_error("missing or duplicate joint name");
  result[i]=static_cast<size_t>(it-trajectory_names.begin());
 } return result;
}
}
