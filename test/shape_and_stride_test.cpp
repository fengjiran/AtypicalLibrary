//
// Created by richard on 6/29/25.
//
#include "storage.h"
#include <gtest/gtest.h>

using namespace atp;

static void check_data(const ShapeAndStride& sz,
                       const std::vector<int64_t>& shape,
                       const std::vector<int64_t>& strides) {
    EXPECT_EQ(shape.size(), strides.size());
    EXPECT_EQ(sz.size(), shape.size());

    int idx = 0;
    for (auto x: shape) {
        EXPECT_EQ(sz.shape_at_uncheck(idx), x);
        EXPECT_EQ(sz.shape_at(idx), x);
        EXPECT_EQ(sz.shape_data()[idx], x);
        EXPECT_EQ(*(sz.shape_begin() + idx), x);
        ++idx;
    }
    EXPECT_EQ(sz.get_shape(), shape);

    idx = 0;
    for (auto x: strides) {
        EXPECT_EQ(sz.stride_at_uncheck(idx), x);
        EXPECT_EQ(sz.stride_at(idx), x);
        EXPECT_EQ(sz.stride_data()[idx], x);
        EXPECT_EQ(*(sz.stride_begin() + idx), x);
        ++idx;
    }
    EXPECT_EQ(sz.get_strides(), strides);
}

TEST(ShapeAndStrideTest, DefaultConstructor) {
    ShapeAndStride sz;
    check_data(sz, {0}, {1});
}

TEST(ShapeAndStrideTest, SetSizes) {
    ShapeAndStride sz;
    sz.set_shape({5, 6, 7, 8});
    check_data(sz, {5, 6, 7, 8}, {1, 0, 0, 0});
}

TEST(ShapeAndStrideTest, ReSizes) {
    ShapeAndStride sz;
    sz.resize(2);

    // small to small growing
    check_data(sz, {0, 0}, {1, 0});

    // small to small growing
    sz.resize(5);
    check_data(sz, {0, 0, 0, 0, 0}, {1, 0, 0, 0, 0});

    for (int i = 0; i < sz.size(); ++i) {
        sz.shape_at_uncheck(i) = i + 1;
        sz.stride_at_uncheck(i) = 2 * (i + 1);
    }

    check_data(sz, {1, 2, 3, 4, 5}, {2, 4, 6, 8, 10});

    // small to small shrinking
    sz.resize(4);
    check_data(sz, {1, 2, 3, 4}, {2, 4, 6, 8});

    // small to small with no size change
    sz.resize(4);
    check_data(sz, {1, 2, 3, 4}, {2, 4, 6, 8});

    // Small to small, growing back so that we can confirm that our "new"
    // data really does get zeroed.
    sz.resize(5);
    check_data(sz, {1, 2, 3, 4, 0}, {2, 4, 6, 8, 0});

    // small to big
    sz.resize(6);
    check_data(sz, {1, 2, 3, 4, 0, 0}, {2, 4, 6, 8, 0, 0});
    sz.shape_at_uncheck(5) = 6;
    sz.stride_at_uncheck(5) = 12;
    check_data(sz, {1, 2, 3, 4, 0, 6}, {2, 4, 6, 8, 0, 12});

    // big to big, growing
    sz.resize(7);
    check_data(sz, {1, 2, 3, 4, 0, 6, 0}, {2, 4, 6, 8, 0, 12, 0});

    sz.shape_at_uncheck(6) = 11;
    sz.stride_at_uncheck(6) = 22;
    check_data(sz, {1, 2, 3, 4, 0, 6, 11}, {2, 4, 6, 8, 0, 12, 22});

    // big to big, shrinking
    sz.resize(6);
    check_data(sz, {1, 2, 3, 4, 0, 6}, {2, 4, 6, 8, 0, 12});

    // Grow back to make sure "new" elements get zeroed in big mode too.
    sz.resize(7);
    check_data(sz, {1, 2, 3, 4, 0, 6, 0}, {2, 4, 6, 8, 0, 12, 0});

    for (int i = 0; i < sz.size(); ++i) {
        sz.shape_at_uncheck(i) = i - 1;
        sz.stride_at_uncheck(i) = 2 * (i - 1);
    }

    check_data(sz, {-1, 0, 1, 2, 3, 4, 5}, {-2, 0, 2, 4, 6, 8, 10});
    sz.resize(5);
    check_data(sz, {-1, 0, 1, 2, 3}, {-2, 0, 2, 4, 6});
}

