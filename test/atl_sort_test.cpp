//
// Created by richard on 10/22/24.
//

#include "atl_construct.h"
#include "atl_sort.hpp"
#include <gtest/gtest.h>
#include <thread>
#include <utils.hpp>

TEST(SortTest, selection) {
    std::vector<int> v{8, 4, 5, 9};
    auto cmp = atp::Iter_less();
    atp::Selection::sort(v.begin(), v.end(), cmp);
    // atp::Selection::sort(v);
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

TEST(SortTest, Shell) {
    std::vector<int> v{8, 4, 5, 9};
    atp::Shell::sort(v.begin(), v.end());
    atp::Sort::Show(v.begin(), v.end());
    EXPECT_TRUE(atp::Sort::IsSorted(v.begin(), v.end()));
}

TEST(SortTest, merge) {
    std::vector<int> v{1, 4, 10, 9};
    auto mid = v.begin() + (v.end() - v.begin()) / 2;
    // atp::MoveMedianOfThree(v.begin(), v.begin(), mid, v.end() - 1, atp::Iter_less());
    // atp::MergeSort::sort(v);
    atp::MergeSort::sort(v.begin(), v.end());
    atp::Sort::Show(v.begin(), v.end());
    EXPECT_TRUE(atp::Sort::IsSorted(v.begin(), v.end()));
}

TEST(SortTest, sort_compare) {
    // GTEST_SKIP();
    int T = 10;
    int N = 1000000;
    auto t0 = atp::SortPerf<atp::StdSort>::Evaluate(T, N);
    // auto t1 = atp::SortPerf<atp::Selection>::Evaluate(T, N);
    // auto t2 = atp::SortPerf<atp::Insertion>::Evaluate(T, N);
    auto t3 = atp::SortPerf<atp::Shell>::Evaluate(T, N);
    auto t4 = atp::SortPerf<atp::MergeSort>::Evaluate(T, N);
    // std::cout << "Selection sort time: " << t1 << "s.\n";
    // std::cout << "Insertion sort time: " << t2 << "s.\n";
    std::cout << "std sort time: " << t0 << "s.\n";
    std::cout << "Shell sort time: " << t3 << "s.\n";
    std::cout << "merge sort time: " << t4 << "s.\n";
}
