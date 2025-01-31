//
// Created by 赵丹 on 25-1-21.
//

#include <gtest/gtest.h>

#include "runtime/shape_tuple.h"

TEST(ObjectTest, object) {
    litetvm::runtime::IntTuple t = {10, 3, 256, 356};
    std::cout << t << std::endl;
}
