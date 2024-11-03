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

class Shell : public Sort {
public:
    template<typename iterator>
    static void sort(iterator begin, iterator end) {
        if (begin == end) {
            return;
        }

        auto n = std::distance(begin, end);
        int h = 1;
        while (h < n / 3) {
            h = 3 * h + 1;
        }

        while (h >= 1) {
            for (auto it = begin + h; it != end; ++it) {
                auto it1 = it;
                while (it1 >= begin + h && *it1 < *(it1 - h)) {
                    std::iter_swap(it1, it1 - h);
                    it1 -= h;
                }
            }
            h = h / 3;
        }
    }

    template<typename iterator>
    static void sort1(iterator begin, iterator end) {
        if (begin == end) {
            return;
        }

        auto n = std::distance(begin, end);
        std::vector<int> hvec{1};
        while (true) {
            int t = 3 * hvec.back() + 1;
            if (t >= n / 3) {
                break;
            }
            hvec.push_back(t);
        }

        int m = static_cast<int>(hvec.size());
        for (int i = m - 1; i >= 0; --i) {
            int h = hvec[i];
            for (auto it = begin + h; it != end; ++it) {
                auto it1 = it;
                while (it1 >= begin + h && *it1 < *(it1 - h)) {
                    std::iter_swap(it1, it1 - h);
                    it1 -= h;
                }
            }
        }
    }

    template<typename T>
    static void sort(std::vector<T>& nums) {
        auto n = nums.size();
        int h = 1;
        while (h < n / 3) {
            h = 3 * h + 1;
        }

        while (h >= 1) {
            for (size_t i = h; i < n; ++i) {
                auto j = i;
                while (j >= h && nums[j] < nums[j - h]) {
                    std::swap(nums[j], nums[j - h]);
                    j -= h;
                }
            }

            h = h / 3;
        }
    }
};

// original merge sort
class MergeSortV1 : public Sort {
public:
    template<typename T>
    static void sort(std::vector<T>& nums) {
        auto n = nums.size();
        std::vector<T> aux(n);
        sort(nums, aux, 0, n - 1);
    }

    template<typename iterator>
    static void sort(iterator begin, iterator end) {
        using T = typename std::iterator_traits<iterator>::value_type;
        std::vector<T> aux(std::distance(begin, end));

    }

private:
    template<typename T>
    static void sort(std::vector<T>& nums, std::vector<T>& aux, int left, int right) {
        if (left >= right) {
            return;
        }

        int mid = left + (right - left) / 2;
        sort(nums, aux, left, mid);
        sort(nums, aux, mid + 1, right);
        merge(nums, aux, left, mid, right);
    }

    template<typename T>
    static void merge(std::vector<T>& nums, std::vector<T>& aux, int left, int mid, int right) {
        for (int i = left; i <= right; ++i) {
            aux[i] = nums[i];
        }

        int i = left;
        int j = mid + 1;
        for (int k = left; k <= right; ++k) {
            if (i > mid) {
                nums[k] = aux[j++];
            } else if (j > right) {
                nums[k] = aux[i++];
            } else if (aux[i] > aux[j]) {
                nums[k] = aux[j++];
            } else {
                nums[k] = aux[i++];
            }
        }
    }

};

// merge sort with improvements
class MergeSortV2 : public Sort {
public:
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