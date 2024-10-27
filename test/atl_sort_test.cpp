//
// Created by richard on 10/22/24.
//

#include "atl_sort.hpp"

#include <gtest/gtest.h>
#include <utils.hpp>
#include <thread>

TEST(SortTest, selection) {
    std::vector<int> v{8, 4, 5, 9};
    // atp::Selection::sort(v.begin(), v.end());
    atp::Selection::sort(v);
    atp::Selection::Show(v.begin(), v.end());
    EXPECT_TRUE(atp::Selection::IsSorted(v.begin(), v.end()));
}

TEST(SortTest, insertion) {
    std::vector<int> v{8, 4, 5, 9};
    // atp::Insertion::sort(v);
    atp::Insertion::sort(v.begin(), v.end());
    atp::Insertion::Show(v.begin(), v.end());
    EXPECT_TRUE(atp::Insertion::IsSorted(v.begin(), v.end()));
}
