//
// Created by richard on 4/29/24.
//
#include "glog/logging.h"
#include "gtest/gtest.h"
// #include "LeakDetector.h"
#include "memory.h"

int main() {
    // auto* ptr = new int(10);
    // // std::cout << "a = " << *ptr << std::endl;
    // delete ptr;
    int a = 1;
    auto b = static_cast<uint16_t>(-a);
    std::cout << b << std::endl;
    testing::InitGoogleTest();
    google::InitGoogleLogging("openXAE_test");
    FLAGS_logtostderr = true;
    return RUN_ALL_TESTS();
}