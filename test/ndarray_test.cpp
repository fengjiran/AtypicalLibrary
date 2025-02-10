//
// Created by richard on 2/9/25.
//
#include <gtest/gtest.h>
#include "runtime/ndarray.h"

using namespace litetvm::runtime;

TEST(NDArrayTest, Empty) {
    auto array = NDArray::Empty({5, 10}, DataType::Float(32), {DLDeviceType::kDLCPU});
    CHECK_EQ(array.use_count(), 1);
    CHECK(array.IsContiguous());
    std::cout << array.Shape() << std::endl;

    auto* t = array.ToDLPack();
    CHECK_EQ(array.use_count(), 2);
    t->deleter(t);
}