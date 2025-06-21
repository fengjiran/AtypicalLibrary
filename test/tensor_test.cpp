//
// Created by 赵丹 on 25-6-17.
//
#include "tensor.h"
#include "fmt/format.h"

#include <gtest/gtest.h>

using namespace atp;

TEST(Tensor, format) {
    fmt::print("hello world\n");
    std::string s1 = fmt::format("The answer is {}.", 42);
    double a = 10.1;
    auto* p = reinterpret_cast<float*>(&a);
    std::cout << "p = " << *p << std::endl;
}

TEST(Tensor, scalar) {
    Scalar s1 = false;
    EXPECT_EQ(s1.toBool(), false);
    EXPECT_TRUE(s1.type() == DLDataTypeCode::kBool);
    EXPECT_TRUE(s1.isBool());
    s1 = true;

    Scalar s2 = 10;
    EXPECT_EQ(s2.toInt(), 10);
    EXPECT_TRUE(s2.isIntegral());

    Scalar s3 = 1.0;
    EXPECT_EQ(s3.toFloat(), 1.0);
    EXPECT_TRUE(s3.isFloatingPoint());
}

TEST(Tensor, base1) {
    Tensor t({3, 10});
    auto t1 = t;
    EXPECT_TRUE(t.shape() == std::vector<int64_t>({3, 10}));
    EXPECT_EQ(t.use_count(), 2);
    EXPECT_TRUE(t1.defined());
}

TEST(Tensor, random) {
    std::vector<int64_t> shape({10, 3, 64, 64});
    int64_t numel = 1;
    for (int64_t x: shape) {
        numel *= x;
    }
    auto t = Tensor::rand(shape);
    EXPECT_EQ(t.numel(), numel);
    EXPECT_EQ(t.nbytes(), numel * 4);
    std::cout << static_cast<float*>(t.data())[0] << std::endl;
    const float* p = t.const_data_ptr<float>();
    std::cout << p[0] << std::endl;
}
