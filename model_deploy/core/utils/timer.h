#ifndef UTILS_TIMER_H_
#define UTILS_TIMER_H_

#include <chrono>

namespace wenet {

class Timer {
 public:
  Timer() : time_start_(std::chrono::steady_clock::now()) {}
  void Reset() { time_start_ = std::chrono::steady_clock::now(); }
  // return int in milliseconds
  int Elapsed() const {
    auto time_now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(time_now -
                                                                 time_start_)
        .count();
  }

 private:
  std::chrono::time_point<std::chrono::steady_clock> time_start_;
};
}  // namespace wenet

#endif  // UTILS_TIMER_H_