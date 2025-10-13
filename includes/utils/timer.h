#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <functional>

namespace utils {

	template <typename Duration = std::chrono::milliseconds>
    class Timer {
    public:
        static unsigned long long measure(const std::function<void()>& func) {
            auto start = std::chrono::high_resolution_clock::now();
            func();
            auto end = std::chrono::high_resolution_clock::now();

            Duration elapsed = std::chrono::duration_cast<Duration>(end - start);
            return static_cast<unsigned long long>(elapsed.count());
        }
    };

} // namespace utils

#endif
