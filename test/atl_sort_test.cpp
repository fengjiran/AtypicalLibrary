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

TEST(SortTest, sort_compare) {
    int T = 100;
    int N = 10000;
    auto t1 = atp::SortPerf<atp::Selection>::Evaluate(T, N);
    auto t2 = atp::SortPerf<atp::Insertion>::Evaluate(T, N);
    std::cout << "Selection time: " << t1 << "s.\n";
    std::cout << "Insertion time: " << t2 << "s.\n";
}
