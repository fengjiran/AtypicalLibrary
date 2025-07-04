//
// Created by 赵丹 on 25-6-17.
//
#include "tensor.h"
#include <fmt/format.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

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
    EXPECT_TRUE(t.defined());
    EXPECT_EQ(t.shape(), shape);
    // EXPECT_TRUE();
    EXPECT_EQ(t.ndim(), 4);
    EXPECT_EQ(t.numel(), numel);
    EXPECT_EQ(t.nbytes(), numel * 4);
    EXPECT_EQ(t.use_count(), 1);
    EXPECT_TRUE(t.unique());
    EXPECT_TRUE(t.dtype() == DataType::Make<float>());
    EXPECT_TRUE(t.device() == DeviceType::kCPU);
    EXPECT_FLOAT_EQ(t.const_data_ptr<float>()[0], static_cast<const float*>(t.const_data_ptr())[0]);

    torch::Tensor t1;
    EXPECT_FALSE(t1.defined());
    EXPECT_EQ(t1.numel(), 0);
    EXPECT_TRUE(t1.is_contiguous());
}
