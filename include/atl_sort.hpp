#ifndef ATL_SORT_HPP
#define ATL_SORT_HPP

#include "utils.hpp"
#include <algorithm>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

namespace atp {

class Sort {
public:
    template<typename iterator>
    static void Show(iterator begin, iterator end) {
        auto size = std::distance(begin, end);
        for (auto it = begin; it != end; ++it) {
            std::cout << *it << (--size ? ", " : "");
        }
        std::cout << std::endl;
    }

    template<typename iterator>
    static bool IsSorted(iterator begin, iterator end) {
        if (begin == end) {
            return true;
        }

        for (auto it = begin + 1; it != end; ++it) {
            if (*(it - 1) > *it) {
                return false;
            }
        }
        return true;
    }
};

class Selection : public Sort {
public:
    template<typename iterator>
    static void sort(iterator begin, iterator end) {
        for (auto it = begin; it != end; ++it) {
            auto minIter = std::min_element(it, end);
            std::iter_swap(it, minIter);
        }
    }

    template<typename T>
    static void sort(std::vector<T>& nums) {
        auto n = nums.size();
        for (size_t i = 0; i < n; ++i) {
            auto minIdx = i;
            for (size_t j = i + 1; j < n; ++j) {
                if (nums[j] < nums[minIdx]) {
                    minIdx = j;
                }
            }
            std::swap(nums[i], nums[minIdx]);
        }
    }
};

class Insertion : public Sort {
public:
    template<typename iterator>
    static void sort(iterator begin, iterator end, bool useSentinel = true) {
        if (useSentinel) {
            sortWithSentinel(begin, end);
        } else {
            sortWithoutSentinel(begin, end);
        }
    }

    template<typename iterator>
    static void sortWithSentinel(iterator begin, iterator end) {
        if (begin == end) {
            return;
        }

        size_t exchNum = 0;
        for (auto it = end - 1; it != begin; --it) {
            if (*it < *(it - 1)) {
                std::iter_swap(it, it - 1);
                exchNum++;
            }
        }

        if (exchNum == 0 || begin + 1 == end) {
            return;
        }

        for (auto it = begin + 2; it != end; ++it) {
            auto x = *it;
            auto it1 = it;
            while (x < *(it1 - 1)) {
                *it1 = *(it1 - 1);
                --it1;
            }
            *it1 = x;
        }
    }

    template<typename iterator>
    static void sortWithoutSentinel(iterator begin, iterator end) {
        if (begin == end) {
            return;
        }

        for (auto it = begin + 1; it != end; ++it) {
            auto x = *it;
            auto it1 = it;
            while (it1 != begin && x < *(it1 - 1)) {
                *it1 = *(it1 - 1);
                --it1;
            }
            *it1 = x;
        }
    }

    template<typename T>
    static void sort(std::vector<T>& nums, bool useSentinel = true) {
        if (useSentinel) {
            sortWithSentinel(nums);
        } else {
            sortWithoutSentinel(nums);
        }
    }

    // with sentinel
    template<typename T>
    static void sortWithSentinel(std::vector<T>& nums) {
        auto n = nums.size();
        size_t exchNum = 0;
        for (size_t i = n - 1; i > 0; --i) {
            if (nums[i] < nums[i - 1]) {
                std::swap(nums[i], nums[i - 1]);
                exchNum++;
            }
        }

        if (exchNum == 0) {
            return;
        }

        for (size_t i = 2; i < n; ++i) {
            auto x = nums[i];
            auto j = i;
            while (x < nums[j - 1]) {
                nums[j] = nums[j - 1];
                --j;
            }
            nums[j] = x;
        }
    }

    // without sentinel
    template<typename T>
    static void sortWithoutSentinel(std::vector<T>& nums) {
        auto n = nums.size();

        for (size_t i = 1; i < n; ++i) {
            auto x = nums[i];
            auto j = i;
            while (j > 0 && x < nums[j - 1]) {
                nums[j] = nums[j - 1];
                --j;
            }
            nums[j] = x;
        }
    }
};

template<typename sortAlgo>
class SortPerf {
public:
    static double Evaluate(int T, int N) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0, 1);

        std::vector<float> nums(N);
        double total = 0;

        for (int t = 0; t < T; ++t) {
            for (int i = 0; i < N; ++i) {
                nums[i] = dist(gen);
            }
            total += time(nums.begin(), nums.end());
        }
        return total;
    }

    template<typename iterator>
    static double time(iterator begin, iterator end) {
        Timer t;
        sortAlgo::sort(begin, end);
        return t.GetElapsedTime();
    }
};

}// namespace atp

#endif