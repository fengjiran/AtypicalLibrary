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

    std::thread t1(f);
    std::thread t2(f);

    if (t1.joinable()) {
        t1.join();
    }

    if (t2.joinable()) {
        t2.join();
    }
}

TEST(MultiThreadTest, test2) {
    auto f = [](int i, int j) {
        std::cout << i + j << std::endl;
    };

    for (int i = 0; i < 4; ++i) {
        std::thread t(f, i, i);
        t.detach();
    }
}