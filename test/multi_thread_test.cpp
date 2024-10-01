//
// Created by richard on 9/30/24.
//
#include <chrono>
#include <gtest/gtest.h>
#include <list>
#include <mutex>
#include <thread>

TEST(MultiThreadTest, test1) {
    auto f1 = [] {
        std::cout << "Hello Concurrent World\n";
        std::cout << "concurrency num: " << std::thread::hardware_concurrency() << std::endl;
    };

    auto f2 = [](int i, int j) {
        std::cout << i + j << std::endl;
    };

    std::thread t1(f1);
    std::thread t2(f1);

    if (t1.joinable()) {
        t1.join();
    }

    if (t2.joinable()) {
        t2.join();
    }

    for (int i = 0; i < 4; ++i) {
        std::thread t(f2, i, i);
        t.detach();
    }
}

TEST(MultiThreadTest, test2) {
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
        // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        std::cout << "The value is: " << *p << std::endl;
    };

    int n = 0;
    std::thread t1(f1, n + 1);
    std::thread t2(f2, std::ref(n));
    std::thread t3(std::move(t2));

    // int x = 1;
    int* x = new int(1);
    std::thread t4(f3, x);
    // delete x;

    if (t1.joinable()) {
        t1.join();
    }

    if (t3.joinable()) {
        t3.join();
    }

    if (t4.joinable()) {
        // t4.detach();
        t4.join();
    }
    delete x;

    std::cout << "Final value of n is " << n << std::endl;
}

TEST(MultiThreadTest, test3) {
    int data = 0;
    std::mutex mtx;

    auto f = [&data, &mtx] {
        for (int i = 0; i < 5000; ++i) {
            mtx.lock();
            data++;
            mtx.unlock();
        }
    };

    std::thread t1(f);
    std::thread t2(f);
    t1.join();
    t2.join();
    // std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::cout << "data = " << data << std::endl;
}
