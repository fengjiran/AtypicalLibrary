#ifndef ATL_SORT_HPP
#define ATL_SORT_HPP

#include "atl_tempbuf.h"
#include "atl_compare_ops.h"
#include "utils.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

namespace atp {

struct Iter_less {
    template<typename iterator1, typename iterator2>
    constexpr bool operator()(iterator1 it1, iterator2 it2) const {
        return *it1 < *it2;
    }
};

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Compare>
void MoveMergeAdaptive(InputIterator1 first1, InputIterator1 last1,
                       InputIterator2 first2, InputIterator2 last2,
                       OutputIterator result,
                       Compare cmp) {
    while (first1 != last1 && first2 != last2) {
        if (cmp(first2, first1)) {
            *result = std::move(*first2);
            ++first2;
        } else {
            *result = std::move(*first1);
            ++first1;
        }
        ++result;
    }

    if (first1 != last1) {
        std::move(first1, last1, result);
    }
}

/// This is a helper function for the merge routines.
template<typename BidirectionalIterator1,
         typename BidirectionalIterator2,
         typename BidirectionalIterator3,
         typename Compare>
void MoveMergeAdaptiveBackward(BidirectionalIterator1 first1,
                               BidirectionalIterator1 last1,
                               BidirectionalIterator2 first2,
                               BidirectionalIterator2 last2,
                               BidirectionalIterator3 result,
                               Compare cmp) {
    if (first1 == last1) {
        std::move_backward(first2, last2, result);
        return;
    }

    if (first2 == last2) {
        return;
    }

    --last1;
    --last2;
    while (first1 != last1 && first2 != last2) {
        if (cmp(last2, last1)) {
            *--result = std::move(*last1);
            --last1;
        } else {
            *--result = std::move(*last2);
            --last2;
        }
    }

    if (first2 != last2) {
        std::move_backward(first2, ++last2, result);
    }
}

/// This is a helper function for the merge routines.
template<typename BidirectionalIterator, typename Distance, typename Pointer, typename Compare>
void MergeAdaptive(BidirectionalIterator first,
                   BidirectionalIterator mid,
                   BidirectionalIterator last,
                   Distance len1,
                   Distance len2,
                   Pointer buffer,
                   Compare cmp) {
    if (len1 <= len2) {
        Pointer bufferEnd = std::move(first, mid, buffer);
        MoveMergeAdaptive(buffer, bufferEnd, mid, last, first, cmp);
    } else {
        Pointer bufferEnd = std::move(mid, last, buffer);
        MoveMergeAdaptiveBackward(first, mid, buffer, bufferEnd, last, cmp);
    }
}

template<typename BidirectionalIterator, typename Compare>
void InplaceMerge(BidirectionalIterator first,
                  BidirectionalIterator mid,
                  BidirectionalIterator last,
                  Compare cmp) {
    using value_type = typename std::iterator_traits<BidirectionalIterator>::value_type;
    using distance_type = typename std::iterator_traits<BidirectionalIterator>::difference_type;
    using buffer_type = TemporaryBuffer<BidirectionalIterator, value_type>;

    if (first == mid || mid == last) {
        return;
    }

    const distance_type len1 = std::distance(first, mid);
    const distance_type len2 = std::distance(mid, last);

    // merge_adaptive will use a buffer for the smaller of
    // [first,middle) and [middle,last).
    buffer_type buf(first, std::min(len1, len2));
    MergeAdaptive(first, mid, last, len1, len2, buf.begin(), cmp);
}

template<typename Iterator, typename Compare>
void MoveMedianOfThree(Iterator result, Iterator first, Iterator mid, Iterator last, Compare cmp) {
    if (cmp(mid, first)) {
        std::iter_swap(first, mid);
    }

    if (cmp(last, mid)) {
        std::iter_swap(mid, last);
    }

    if (cmp(mid, first)) {
        std::iter_swap(first, mid);
    }

    std::iter_swap(result, mid);
}

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
        return IsSorted(begin, end, Iter_less_iter());
    }

    template<typename iterator, typename Compare>
    static bool IsSorted(iterator begin, iterator end, Compare cmp) {
        if (begin == end) {
            return true;
        }

        for (auto it = begin + 1; it != end; ++it) {
            if (cmp(it, it - 1)) {
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
        sort(begin, end, Iter_less());
    }

    template<typename iterator, typename Compare>
    static void sort(iterator begin, iterator end, Compare cmp) {
        for (auto it = begin; it != end; ++it) {
            auto it1 = it;
            for (auto it2 = it + 1; it2 != end; ++it2) {
                if (cmp(it2, it1)) {
                    it1 = it2;
                }
            }

            std::iter_swap(it, it1);
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
    static void sort(iterator begin, iterator end) {
        sort(begin, end, Iter_less());
    }

    template<typename iterator, typename Compare>
    static void sort(iterator begin, iterator end, Compare cmp, bool useSentinel = true) {
        if (useSentinel) {
            sortWithSentinel(begin, end, cmp);
        } else {
            sortWithoutSentinel(begin, end, cmp);
        }
    }

    template<typename iterator, typename Compare>
    static void sortWithSentinel(iterator begin, iterator end, Compare cmp) {
        if (begin == end || begin + 1 == end) {
            return;
        }

        size_t exchNum = 0;
        for (auto it = end - 1; it != begin; --it) {
            if (cmp(it, it - 1)) {
                std::iter_swap(it, it - 1);
                exchNum++;
            }
        }

        if (exchNum == 0) {
            return;
        }

        for (auto it = begin + 2; it != end; ++it) {
            auto x = it;
            auto it1 = it;
            while (cmp(x, it1 - 1)) {
                *it1 = *(it1 - 1);
                --it1;
            }
            *it1 = *x;
        }
    }

    template<typename iterator, typename Compare>
    static void sortWithoutSentinel(iterator begin, iterator end, Compare cmp) {
        if (begin == end || begin + 1 == end) {
            return;
        }

        for (auto it = begin + 1; it != end; ++it) {
            auto x = *it;
            auto it1 = it;
            while (it1 != begin && *(it1 - 1) > x) {
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
        sort(begin, end, Iter_less());
    }

    template<typename iterator, typename Compare>
    static void sort(iterator begin, iterator end, Compare cmp) {
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
                while (it1 >= begin + h && cmp(it1, it1 - h)) {
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
class MergeSort : public Sort {
public:
    template<typename iterator>
    static void sort(iterator begin, iterator end) {
        sort(begin, end, Iter_less());
    }

    template<typename iterator, typename Compare>
    static void sort(iterator begin, iterator end, Compare cmp, int cutoff = 15) {
        // if (begin + 1 >= end) {
        //     return;
        // }

        if (std::distance(begin, end) <= cutoff) {
            Insertion::sort(begin, end, cmp);
            return;
        }

        auto mid = begin + (end - begin) / 2;
        sort(begin, mid);
        sort(mid, end);
        if (!cmp(mid, mid - 1)) {
            return;
        }

        // if (*(mid - 1) <= *mid) {
        //     return;
        // }
        InplaceMerge(begin, mid, end, cmp);
        // std::inplace_merge(begin, mid, end);
    }

    template<typename T>
    static void sort(std::vector<T>& nums) {
        auto n = nums.size();
        std::vector<T> aux(n);
        sort(nums, aux, 0, n - 1);
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

class QuickSort : public Sort {
public:
    template<typename RandomAccessIterator>
    void static sort(RandomAccessIterator begin, RandomAccessIterator end) {
        //
    }
};


class StdSort : public Sort {
public:
    template<typename iterator>
    static void sort(iterator begin, iterator end) {
        sort(begin, end, Iter_less());
    }

    template<typename iterator, typename Compare>
    static void sort(iterator begin, iterator end, Compare cmp) {
        std::sort(begin, end);
    }
};

template<typename SortAlgo>
class SortPerf {
public:
    static double Evaluate(int T, int N) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0, 1);

        auto cmp = Iter_less();

        std::vector<float> nums(N);
        double total = 0;

        for (int t = 0; t < T; ++t) {
            for (int i = 0; i < N; ++i) {
                nums[i] = dist(gen);
            }
            total += time(nums.begin(), nums.end(), cmp);
        }
        assert(Sort::IsSorted(nums.begin(), nums.end()));
        return total;
    }

    template<typename iterator, typename Compare>
    static double time(iterator begin, iterator end, Compare cmp) {
        Timer t;
        SortAlgo::sort(begin, end, cmp);
        return t.GetElapsedTime();
    }
};

}// namespace atp

#endif