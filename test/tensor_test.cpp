//
// Created by 赵丹 on 25-6-17.
//
#include "tensor.h"

#include <gtest/gtest.h>

using namespace atp;

TEST(Tensor, base1) {
    Tensor t({3, 10});
    auto t1 = t;
    EXPECT_TRUE(t.shape() == std::vector<int64_t>({3, 10}));
    EXPECT_EQ(t.use_count(), 2);
}
