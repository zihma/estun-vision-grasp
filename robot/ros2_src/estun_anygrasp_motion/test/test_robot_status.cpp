#include "robot_status.hpp"
#include <thread>
#include <stdexcept>
#include <iostream>
using namespace std::chrono_literals;
using estun_anygrasp::RobotStatus;
void check(bool value){if(!value)throw std::runtime_error("RobotStatus regression failed");}
int main(){
 RobotStatus state;check(!state.ready());check(!state.waitForFirst(5ms));check(state.describe()=="no RobotInfo received");
 std::thread publisher([&]{std::this_thread::sleep_for(30ms);state.update("AutoReady",0);});
 bool got=state.waitForFirst(500ms);publisher.join();check(got);check(state.ready());
 state.update("AutoReady",1);check(!state.ready());check(state.describe().find("fault=1")!=std::string::npos);
 state.update("Idle",0);check(!state.ready());
 state.update("AutoReady",0);check(state.ready());std::this_thread::sleep_for(2050ms);check(!state.ready());
 std::cout<<"Delayed first message accepted; missing, fault, wrong mode and stale feedback rejected\n";
}
