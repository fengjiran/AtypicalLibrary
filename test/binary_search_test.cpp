//
// Created by richard on 7/31/24.
//

#include "binary_search.h"
#include "gtest/gtest.h"

TEST(BinarySearch, t1) {
    std::vector<int> nums1{2, 4, 5, 6, 9};
    EXPECT_EQ(BinarySearch<int>::FindFirstGEItem(nums1, 0, nums1.size() - 1, 2), 0);
    EXPECT_EQ(BinarySearch<int>::FindFirstGEItem(nums1, 0, nums1.size() - 1, 1), 0);
    EXPECT_EQ(BinarySearch<int>::FindFirstGEItem(nums1, 0, nums1.size() - 1, 9), 4);
    EXPECT_EQ(BinarySearch<int>::FindFirstGEItem(nums1, 0, nums1.size() - 1, 10), -1);

    EXPECT_EQ(BinarySearch<int>::FindFirstGTItem(nums1, 0, nums1.size() - 1, 2), 1);
    EXPECT_EQ(BinarySearch<int>::FindFirstGTItem(nums1, 0, nums1.size() - 1, 1), 0);
    EXPECT_EQ(BinarySearch<int>::FindFirstGTItem(nums1, 0, nums1.size() - 1, 9), -1);

    EXPECT_EQ(BinarySearch<int>::LeftBound(nums1, 0, nums1.size() - 1, 4), 1);
    EXPECT_EQ(BinarySearch<int>::LeftBound(nums1, 0, nums1.size() - 1, 2), 0);
    EXPECT_EQ(BinarySearch<int>::LeftBound(nums1, 0, nums1.size() - 1, 9), 4);
    EXPECT_EQ(BinarySearch<int>::LeftBound(nums1, 0, nums1.size() - 1, 1), -1);
    EXPECT_EQ(BinarySearch<int>::LeftBound(nums1, 0, nums1.size() - 1, 10), -1);


    std::vector<int> nums2{2, 3, 3, 4, 5, 6, 9};
    EXPECT_EQ(BinarySearch<int>::FindFirstGEItem(nums2, 0, nums2.size() - 1, 10), -1);

    int a = 10;
    int& b = a;
    static_assert(std::is_same_v<decltype((a)), int&>);
}
