//
// Created by richard on 9/30/24.
//
#include <gtest/gtest.h>
#include <list>
#include <mutex>
#include <thread>

TEST(MultiThreadTest, test1) {
    auto f = []() {
        std::cout << "Hello Concurrent World\n";
        std::cout << "concurrency num: " << std::thread::hardware_concurrency() << std::endl;
    };
}