//
// Created by richard on 4/29/24.
//
#include "glog/logging.h"
#include "gtest/gtest.h"
// #include "LeakDetector.h"
// #include "memory.h"
#include "shape_tuple.h"

int main() {

    litetvm::runtime::IntTuple t = {10, 3, 256, 356};
    std::cout << t << std::endl;

    testing::InitGoogleTest();
    google::InitGoogleLogging("openXAE_test");
    FLAGS_logtostderr = true;
    return RUN_ALL_TESTS();
}