//
// Created by richard on 4/29/24.
//
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "LeakDetector.h"

int main() {
    auto a = new int;
    auto b = new int[10];
    // auto* p = reinterpret_cast<Chunk*>((char*)a - sizeof(Chunk));
    std::cout << a << std::endl;
    std::cout << b << std::endl;

    delete a;
    // delete[] b;
    testing::InitGoogleTest();
    google::InitGoogleLogging("openXAE_test");
    FLAGS_logtostderr = true;
    return RUN_ALL_TESTS();
}