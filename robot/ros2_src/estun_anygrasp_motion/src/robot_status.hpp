#pragma once
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <sstream>
#include <string>
namespace estun_anygrasp {
class RobotStatus {
 public:
  void update(const std::string& state, int fault) {
    {std::lock_guard<std::mutex> lock(mutex_);received_=true;state_=state;fault_=fault;receipt_=Clock::now();}
    changed_.notify_all();
  }
  bool waitForFirst(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    return changed_.wait_for(lock,timeout,[this]{return received_;});
  }
  bool ready() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return received_&&age()<=2.&&state_=="AutoReady"&&fault_==0;
  }
  std::string describe() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if(!received_)return "no RobotInfo received";
    std::ostringstream out;out<<"state='"<<state_<<"' fault="<<fault_<<" age="<<age()<<" s (limit 2 s)";return out.str();
  }
 private:
  using Clock=std::chrono::steady_clock;
  double age() const {return std::chrono::duration<double>(Clock::now()-receipt_).count();}
  mutable std::mutex mutex_;
  std::condition_variable changed_;
  bool received_=false;
  std::string state_;
  int fault_=-1;
  Clock::time_point receipt_{};
};
}
