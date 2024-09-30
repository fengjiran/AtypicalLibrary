//
// Created by richard on 9/30/24.
//
#include <chrono>
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

TEST(MultiThreadTest, test3) {
    auto f1 = [](int n) {
        std::cout << "thread id: " << std::this_thread::get_id() << std::endl;
        for (int i = 0; i < 5; ++i) {
            std::cout << "Thread 1 is executing\n";
            ++n;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    };

    auto f2 = [](int& n) {
        std::cout << "thread id: " << std::this_thread::get_id() << std::endl;
        for (int i = 0; i < 5; ++i) {
            std::cout << "Thread 2 is executing\n";
            ++n;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    };

    auto f3 = [](const int* p) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        std::cout << *p << std::endl;
    };

    int n = 0;

    std::thread t2(f1, n + 1);      // pass by value
    std::thread t3(f2, std::ref(n));// pass by reference
    std::thread t4(std::move(t3));  // t4 is now running f2, t3 is no longer a thread

    int x = 1;
    std::thread t1(f3, &x);
    t1.detach();

    // if (t1.joinable()) {
    //     // t1.join();
    //     t1.detach();
    // }

    if (t2.joinable()) {
        t2.join();
        // t2.detach();
    }

    if (t4.joinable()) {
        t4.join();
        // t4.detach();
    }

    std::cout << "Final value of n is " << n << std::endl;
}