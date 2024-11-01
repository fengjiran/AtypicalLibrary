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


    std::vector<int> nums2{3,5,8,10};
    // EXPECT_EQ(BinarySearch<int>::FindFirstGEItem(nums2, 0, nums2.size() - 1, 3), 2);

    auto f = [](std::vector<int>& nums, int target) {
        int left = 0;
        int right = (int) nums.size();
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        if (left == nums.size()) {
            return (int)nums.size();
        }

        return left;
    };

    int x = 15;
    int right = f(nums2, x);
    int left = right - 1;
    int k = 2;
    while (k--) {
        if (left < 0) {
            right++;
        } else if (right >= nums2.size()) {
            left--;
        } else if (x - nums2[left] <= nums2[right] - x) {
            left--;
        } else {
            right++;
        }
    }

    std::cout << left + 1 << std::endl;
    std::cout << right - 1 << std::endl;
    std::vector<int> y(nums2.begin() + left + 1, nums2.begin() + right);
    for (int t : y) {
        std::cout << t << std::endl;
    }

    int a = 10;
    int& b = a;
    static_assert(std::is_same_v<decltype((a)), int&>);
}
