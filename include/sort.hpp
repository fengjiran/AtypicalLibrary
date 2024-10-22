#ifndef SORT_HPP
#define SORT_HPP

#include <iostream>

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

}// namespace atp

#endif