#ifndef ATL_SORT_HPP
#define ATL_SORT_HPP

#include "atl_compare_ops.h"
#include "atl_heap.h"
#include "atl_tempbuf.h"
#include "utils.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

namespace atp {

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Compare>
void MoveMergeAdaptive(InputIterator1 first1, InputIterator1 last1,
                       InputIterator2 first2, InputIterator2 last2,
                       OutputIterator result,
                       const Compare& cmp) {
    while (first1 != last1 && first2 != last2) {
        if (cmp(*first2, *first1)) {
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
                               const Compare& cmp) {
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
        if (cmp(*last2, *last1)) {
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
                   const Compare& cmp) {
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
                  const Compare& cmp) {
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
void MoveMedian3ToPos(Iterator pos, Iterator first, Iterator second, Iterator third, const Compare& cmp) {
    if (cmp(*second, *first)) {
        std::iter_swap(first, second);
    }

    if (cmp(*third, *second)) {
        std::iter_swap(second, third);
    }

    if (cmp(*second, *first)) {
        std::iter_swap(first, second);
    }

    std::iter_swap(pos, second);
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

    template<typename ForwardIterator, typename Compare>
    static ForwardIterator IsSortedUntil(ForwardIterator begin, ForwardIterator end, Compare cmp) {
        if (begin == end) {
            return end;
        }

        auto next = begin + 1;
        while (next != end) {
            if (cmp(*next, *begin)) {
                return next;
            }
            begin = next;
            ++next;
        }
        return next;
    }

    template<typename iterator>
    static bool IsSorted(iterator begin, iterator end) {
        return IsSortedUntil(begin, end, Val_less_val()) == end;
    }

    template<typename iterator, typename Compare>
    static bool IsSorted(iterator begin, iterator end, const Compare& cmp) {
        return IsSortedUntil(begin, end, cmp) == end;
    }
};

class Selection : public Sort {
public:
    template<typename iterator>
    static void sort(iterator begin, iterator end) {
        sort(begin, end, Val_less_val());
    }

    template<typename iterator, typename Compare>
    static void sort(iterator begin, iterator end, const Compare& cmp) {
        for (auto it = begin; it != end; ++it) {
            auto j = it;
            for (auto it1 = it + 1; it1 != end; ++it1) {
                if (cmp(*it1, *j)) {
                    j = it1;
                }
            }

            std::iter_swap(it, j);
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
    template<typename RandomAccessIterator>
    static void sort(RandomAccessIterator begin, RandomAccessIterator end) {
        sort(begin, end, Val_less_val());
    }

    template<typename RandomAccessIterator, typename Compare>
    static void sort(RandomAccessIterator begin, RandomAccessIterator end, const Compare& cmp) {
        if (begin == end) {
            return;
        }

        for (auto it = begin + 1; it != end; ++it) {
            auto val = std::move(*it);
            if (cmp(val, *begin)) {
                std::move_backward(begin, it, it + 1);
                *begin = std::move(val);
            } else {
                auto j = it;
                while (cmp(val, *(j - 1))) {
                    *j = std::move(*(j - 1));
                    --j;
                }
                *j = std::move(val);
            }
        }
    }

    // with sentinel
    template<typename T>
    static void sort(std::vector<T>& nums) {
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
};

class Shell : public Sort {
public:
    template<typename RandomAccessIterator>
    static void sort(RandomAccessIterator begin, RandomAccessIterator end) {
        sort(begin, end, Val_less_val());
    }

    template<typename RandomAccessIterator, typename Compare>
    static void sort(RandomAccessIterator begin, RandomAccessIterator end, const Compare& cmp) {
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
                auto val = std::move(*it);
                auto j = it;
                while (j >= begin + h && cmp(val, *(j - h))) {
                    *j = std::move(*(j - h));
                    j -= h;
                }
                *j = std::move(val);
            }
            h = h / 3;
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
    enum class Threshold {
        kCutoff = 24
    };

    template<typename iterator>
    static void sort(iterator begin, iterator end) {
        sort(begin, end, Val_less_val());
    }

    template<typename iterator, typename Compare>
    static void sort(iterator begin, iterator end, const Compare& cmp) {
        if (begin + 1 >= end) {
            return;
        }

        auto n = std::distance(begin, end);

        if (n <= static_cast<decltype(n)>(Threshold::kCutoff)) {
            Insertion::sort(begin, end, cmp);
            return;
        }

        auto mid = begin + (end - begin) / 2;
        sort(begin, mid, cmp);
        sort(mid, end, cmp);
        if (cmp(*(mid - 1), *mid)) {
            return;
        }

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

class HeapSort : public Sort {
public:
    template<typename RandomAccessIterator, typename Compare>
    static void sort(RandomAccessIterator begin, RandomAccessIterator end, const Compare& cmp) {
        atp::make_heap(begin, end);
        // assert(atp::is_heap(begin, end));
        while (end - begin > 1) {
            --end;
            std::iter_swap(begin, end);
            sink(begin, 0, end - begin, cmp);
        }
    }

    template<typename RandomAccessIterator>
    static void sort(RandomAccessIterator begin, RandomAccessIterator end) {
        sort(begin, end, Val_less_val());
    }
};

class QuickSort : public Sort {
public:
    enum class Threshold {
        kInsertSortCutoff = 16,
        kMedian3Cutoff = 40
    };

    template<typename RandomAccessIterator>
    static void sort(RandomAccessIterator begin, RandomAccessIterator end) {
        auto cmp = Val_less_val();
        _sort(begin, end, std::log2(end - begin) * 2, cmp);
        Insertion::sort(begin, end, cmp);
        // BentlyMcIlroyQuickSort(begin, end, Val_less_val());
    }

    template<typename RandomAccessIterator, typename Compare>
    static void sort(RandomAccessIterator begin, RandomAccessIterator end, const Compare& cmp) {
        _sort(begin, end, std::log2(end - begin) * 2, cmp);
        Insertion::sort(begin, end, cmp);
        // BentlyMcIlroyQuickSort(begin, end, Val_less_val());
    }

    // tail recursive quick sort
    template<typename RandomAccessIterator, typename Compare>
    static void _sort(RandomAccessIterator begin, RandomAccessIterator end, int depth_limit, const Compare& cmp) {
        using difference_type = typename std::iterator_traits<RandomAccessIterator>::difference_type;
        while (std::distance(begin, end) > static_cast<difference_type>(Threshold::kInsertSortCutoff)) {
            if (depth_limit == 0) {
                HeapSort::sort(begin, end, cmp);
                return;
            }
            --depth_limit;

            auto cut = partition(begin, end, cmp);
            _sort(cut + 1, end, depth_limit, cmp);
            end = cut;
        }

        // if (n > static_cast<decltype(n)>(Threshold::kInsertSortCutoff)) {
        //     // Insertion::sort(begin, end, cmp);
        //     // return;
        //     auto cut = partition(begin, end, cmp);
        //     _sort(begin, cut, cmp);
        //     _sort(cut + 1, end, cmp);
        // }
    }

    // template<typename RandomAccessIterator, typename Compare>
    // static void sort(RandomAccessIterator begin, RandomAccessIterator end, const Compare& cmp) {
    //     auto n = std::distance(begin, end);
    //     if (n <= static_cast<decltype(n)>(Threshold::kInsertSortCutoff)) {
    //         Insertion::sort(begin, end, cmp);
    //         return;
    //     }
    //
    //     auto cut = partition(begin, end, cmp);
    //     sort(begin, cut, cmp);
    //     sort(cut + 1, end, cmp);
    // }

    template<typename RandomAccessIterator, typename Compare>
    static RandomAccessIterator partition(RandomAccessIterator begin, RandomAccessIterator end, const Compare& cmp) {
        auto n = std::distance(begin, end);
        auto mid = begin + n / 2;

        if (n <= static_cast<decltype(n)>(Threshold::kMedian3Cutoff)) {// use median-of-3 as partitioning element
            MoveMedian3ToPos(begin, begin + 1, mid, end - 1, cmp);
        } else {// use Tukey ninther as partitioning element
            auto eps = n / 8;
            MoveMedian3ToPos(begin, begin, begin + eps, begin + eps + eps, cmp);
            MoveMedian3ToPos(mid, mid - eps, mid, mid + eps, cmp);
            MoveMedian3ToPos(end - 1, end - 1 - eps - eps, end - 1 - eps, end - 1, cmp);

            MoveMedian3ToPos(begin, begin, mid, end - 1, cmp);
        }

        auto i = begin;
        auto j = end;
        while (true) {
            while (cmp(*++i, *begin)) {}
            while (cmp(*begin, *--j)) {}
            if (!(i < j)) {
                break;
            }
            std::iter_swap(i, j);
        }
        std::iter_swap(begin, j);
        return j;
    }

    template<typename RandomAccessIterator, typename Compare>
    static void BentlyMcIlroyQuickSort(RandomAccessIterator begin, RandomAccessIterator end, const Compare& cmp) {
        auto n = std::distance(begin, end);
        if (n <= static_cast<decltype(n)>(Threshold::kInsertSortCutoff)) {
            Insertion::sort(begin, end, cmp);
            return;
        }

        auto mid = begin + n / 2;

        if (n <= static_cast<decltype(n)>(Threshold::kMedian3Cutoff)) {// use median-of-3 as partitioning element
            MoveMedian3ToPos(begin, begin + 1, mid, end - 1, cmp);
        } else {// use Tukey ninther as partitioning element
            auto eps = n / 8;
            MoveMedian3ToPos(begin, begin, begin + eps, begin + eps + eps, cmp);
            MoveMedian3ToPos(mid, mid - eps, mid, mid + eps, cmp);
            MoveMedian3ToPos(end - 1, end - 1 - eps - eps, end - 1 - eps, end - 1, cmp);
            MoveMedian3ToPos(begin, begin, mid, end - 1, cmp);
        }

        auto i = begin;
        auto j = end;
        auto p = begin;
        auto q = end;
        while (true) {
            while (cmp(*++i, *begin)) {}
            while (cmp(*begin, *--j)) {}

            if (i == j && *i == *begin) {
                std::iter_swap(i, ++p);
            }

            if (!(i < j)) {
                break;
            }
            std::iter_swap(i, j);

            if (*i == *begin) {
                std::iter_swap(i, ++p);
            }

            if (*j == *begin) {
                std::iter_swap(j, --q);
            }
        }

        i = j + 1;
        for (auto k = begin; k != p + 1; ++k) {
            std::iter_swap(k, j--);
        }

        for (auto k = end - 1; k != q - 1; --k) {
            std::iter_swap(k, i++);
        }

        BentlyMcIlroyQuickSort(begin, j + 1, cmp);
        BentlyMcIlroyQuickSort(i, end, cmp);
    }
};

class StdSort : public Sort {
public:
    template<typename iterator>
    static void sort(iterator begin, iterator end) {
        sort(begin, end, Val_less_val());
    }

    template<typename iterator, typename Compare>
    static void sort(iterator begin, iterator end, Compare cmp) {
        std::sort(begin, end, cmp);
    }
};

template<typename SortAlgo>
class SortPerf {
public:
    static double Evaluate(int T, int N) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0, 1);

        std::vector<float> nums(N, 1);
        double total = 0;

        for (int t = 0; t < T; ++t) {
            for (int i = 0; i < N; ++i) {
                nums[i] = dist(gen);
            }
            total += time(nums.begin(), nums.end());
        }
        assert(Sort::IsSorted(nums.begin(), nums.end()));
        return total;
    }

    template<typename iterator>
    static double time(iterator begin, iterator end) {
        Timer t;
        SortAlgo::sort(begin, end);
        return t.GetElapsedTime();
    }
};

}// namespace atp

#endif