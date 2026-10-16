#include "blended.hpp"
#include <iostream>
using namespace estun_anygrasp;
void require(bool v){if(!v)throw std::runtime_error("test failed");}
int main(){
 Joints start{},goal{1.813579,-.75,.4,-.2,1.2,.1};auto w=waypoints(start,goal,2.);
 require(w.size()==19);Joints previous=start;for(auto q:w){require(difference(previous,q)<=.10+1e-9);previous=q;}require(difference(previous,goal)<1e-9);
 require(waypoints(start,start,2.).empty());goal[0]=2.1;bool rejected=false;try{waypoints(start,goal,2.);}catch(...){rejected=true;}require(rejected);
 auto indices=joint_indices({"joint6","joint5","joint4","joint3","joint2","joint1"});require(indices[0]==5&&indices[5]==0);
 rejected=false;try{joint_indices({"joint1","joint1","joint3","joint4","joint5","joint6"});}catch(...){rejected=true;}require(rejected);
 goal[0]=NAN;rejected=false;try{waypoints(start,goal,2.);}catch(...){rejected=true;}require(rejected);
 std::cout<<"19 segments / 20 keypoints, limit, finite and name mapping tests passed\n";
}
