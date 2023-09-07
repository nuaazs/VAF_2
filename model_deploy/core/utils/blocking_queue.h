#ifndef UTILS_BLOCKING_QUEUE_H_
#define UTILS_BLOCKING_QUEUE_H_

#include <condition_variable>
#include <limits>
#include <mutex>
#include <queue>
#include <utility>

namespace wenet {

#define WENET_DISALLOW_COPY_AND_ASSIGN(Type) \
  Type(const Type&) = delete;                \
  Type& operator=(const Type&) = delete;

template <typename T>
class BlockingQueue {
 public:
  explicit BlockingQueue(size_t capacity = std::numeric_limits<int>::max())
      : capacity_(capacity) {}

  void Push(const T& value) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      while (queue_.size() >= capacity_) {
        not_full_condition_.wait(lock);
      }
      queue_.push(value);
    }
    not_empty_condition_.notify_one();
  }

  void Push(T&& value) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      while (queue_.size() >= capacity_) {
        not_full_condition_.wait(lock);
      }
      queue_.push(std::move(value));
    }
    not_empty_condition_.notify_one();
  }

  T Pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (queue_.empty()) {
      not_empty_condition_.wait(lock);
    }
    T t(std::move(queue_.front()));
    queue_.pop();
    not_full_condition_.notify_one();
    return t;
  }

  bool Empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  size_t Size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

  void Clear() {
    while (!Empty()) {
      Pop();
    }
  }

 private:
  size_t capacity_;
  mutable std::mutex mutex_;
  std::condition_variable not_full_condition_;
  std::condition_variable not_empty_condition_;
  std::queue<T> queue_;

 public:
  WENET_DISALLOW_COPY_AND_ASSIGN(BlockingQueue);
};

}  // namespace wenet

#endif  // UTILS_BLOCKING_QUEUE_H_