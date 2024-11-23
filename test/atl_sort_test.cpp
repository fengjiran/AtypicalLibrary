//
// Created by richard on 10/22/24.
//

#include "atl_construct.h"
#include "atl_sort.hpp"

#include <gtest/gtest.h>
#include <queue>
#include <thread>
#include <utils.hpp>

TEST(SortTest, selection) {
    std::vector<int> v1{8, 4, 5, 9};
    atp::Selection::sort(v1.begin(), v1.end());
    atp::Selection::Show(v1.begin(), v1.end());
    EXPECT_TRUE(atp::Selection::IsSorted(v1.begin(), v1.end()));
}

TEST(SortTest, insertion) {
    // GTEST_SKIP();
    std::vector<int> v2{3, 6, 2, 9, 7};
    atp::Insertion::sort(v2.begin(), v2.end());
    atp::Insertion::Show(v2.begin(), v2.end());
    EXPECT_TRUE(atp::Insertion::IsSorted(v2.begin(), v2.end()));
}

TEST(SortTest, Shell) {
    std::vector<int> v3{8, 4, 5, 9};
    atp::Shell::sort(v3.begin(), v3.end());
    atp::Sort::Show(v3.begin(), v3.end());
    EXPECT_TRUE(atp::Sort::IsSorted(v3.begin(), v3.end()));
}

TEST(SortTest, merge) {
    std::vector<int> v{1, 4, 10, 9};
    atp::MergeSort::sort(v.begin(), v.end());
    atp::Sort::Show(v.begin(), v.end());
    EXPECT_TRUE(atp::Sort::IsSorted(v.begin(), v.end()));
}

TEST(SortTest, quick) {
    std::vector<int> v{3, 6, 2, 9, 7};
    // auto mid = v.begin() + (v.end() - v.begin()) / 2;
    // MoveMedianOfThree(v.begin(), v.begin() + 1, mid, v.end(), atp::Val_less_val());
    atp::QuickSort::sort(v.begin(), v.end());
    atp::Sort::Show(v.begin(), v.end());
    EXPECT_TRUE(atp::Sort::IsSorted(v.begin(), v.end()));
}

TEST(SortTest, heap) {
    std::vector<int> v{3, 6, 2, 9, 7};
    atp::make_heap(v.begin(), v.end());
    v.push_back(10);
    atp::push_heap(v.begin(), v.end());
    atp::Sort::Show(v.begin(), v.end());
    EXPECT_TRUE(atp::is_heap(v.begin(), v.end()));

    atp::pop_heap(v.begin(), v.end());
    atp::Sort::Show(v.begin(), v.end());
    EXPECT_FALSE(atp::is_heap(v.begin(), v.end()));
}

TEST(SortTest, heap_sort) {
    std::vector<int> v{3, 6, 2, 9, 7};
    atp::HeapSort::sort(v.begin(), v.end());
    atp::Sort::Show(v.begin(), v.end());
    EXPECT_TRUE(atp::Sort::IsSorted(v.begin(), v.end()));
    std::set<int> a;
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
    auto t5 = atp::SortPerf<atp::QuickSort>::Evaluate(T, N);
    auto t6 = atp::SortPerf<atp::HeapSort>::Evaluate(T, N);

    // std::cout << "Selection sort time: " << t1 << "s.\n";
    // std::cout << "Insertion sort time: " << t2 << "s.\n";
    std::cout << "std sort time: " << t0 << "s.\n";
    std::cout << "Shell sort time: " << t3 << "s.\n";
    std::cout << "merge sort time: " << t4 << "s.\n";
    std::cout << "quick sort time: " << t5 << "s.\n";
    std::cout << "heap sort time: " << t6 << "s.\n";
}
