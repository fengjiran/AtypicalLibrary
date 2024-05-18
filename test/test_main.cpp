//
// Created by richard on 4/29/24.
//
#include "gtest/gtest.h"
#include "glog/logging.h"

int main() {
    testing::InitGoogleTest();
    google::InitGoogleLogging("openXAE_test");
    FLAGS_logtostderr = true;
    return RUN_ALL_TESTS();
}