//
// Created by richard on 10/22/24.
//

#include <gtest/gtest.h>
#include "sort.hpp"

TEST(SortTest, selection) {
    std::vector<int> v{8, 4, 5, 9};
    // atp::Selection::sort(v.begin(), v.end());
    atp::Selection::sort(v);
    atp::Selection::Show(v.begin(), v.end());
    EXPECT_TRUE(atp::Selection::IsSorted(v.begin(), v.end()));
}
