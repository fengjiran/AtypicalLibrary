//
// Created by 赵丹 on 25-6-17.
//
#include "tensor.h"
#include <fmt/format.h>

#include <gtest/gtest.h>

using namespace atp;

TEST(Tensor, format) {
    GTEST_SKIP();
    fmt::print("hello world\n");
    std::string s1 = fmt::format("The answer is {}.", 42);
}

TEST(Tensor, envs) {
    EXPECT_TRUE(has_env("THP_MEM_ALLOC_ENABLE"));
    EXPECT_TRUE(check_env("THP_MEM_ALLOC_ENABLE"));
    // std::cout << sysconf(_SC_PAGESIZE);
}

TEST(Tensor, ShapeAndStride) {
    ShapeAndStride s1;
    EXPECT_EQ(s1.size(), 1);
    EXPECT_EQ(s1.shape_begin()[0], 0);
    EXPECT_EQ(s1.stride_begin()[0], 1);
    EXPECT_TRUE(s1.get_shape() == std::vector<int64_t>{0});
    EXPECT_TRUE(s1.get_strides() == std::vector<int64_t>{1});

    std::vector<int64_t> shape{10, 3, 256, 256};
    s1.set_shape(shape);
    EXPECT_EQ(s1.size(), 4);
    EXPECT_TRUE(s1.get_shape() == shape);
}

TEST(Tensor, unique_void_ptr) {
    struct Context {
        explicit Context(void* ptr) : data(ptr) {}
        void delete_ptr() const {
            // std::cout << "call free.\n";
            free(data);
        }
        void* data;
    };

    auto default_deleter = [](void* ptr) {
        static_cast<Context*>(ptr)->delete_ptr();
    };

    auto* ptr = malloc(10);
    Context ctx(ptr);
    UniqueVoidPtr p(ptr, static_cast<void*>(&ctx), default_deleter);
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
    // auto t = Tensor::randint(0, 10, shape);
    EXPECT_EQ(t.numel(), numel);
    std::cout << t.const_data_ptr<float>()[0] << std::endl;
    std::cout << static_cast<const float*>(t.const_data_ptr())[0] << std::endl;
    // std::cout << static_cast<double*>(t.data())[0] << std::endl;
    // auto* p = t.const_data_ptr<int>();
    // std::cout << p[0] << std::endl;
    // EXPECT_TRUE(std::isfinite(p[0]));
    // EXPECT_TRUE(p[0] != std::ceil(p[0]));
    //
    // auto* p1 = static_cast<double*>(static_cast<void*>(shape.data()));
    // EXPECT_TRUE(std::isfinite(p1[6]));
    // std::cout << p1[0] << std::endl;
    // EXPECT_TRUE(p1[0] != std::ceil(p1[0]));
}
