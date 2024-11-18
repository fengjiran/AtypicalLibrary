//
// Created by richard on 11/17/24.
//

#ifndef ATL_HEAP_H
#define ATL_HEAP_H

#include <iterator>

namespace atp {

template<typename RandomAccessIterator,
         typename Compare,
         typename difference_type = typename std::iterator_traits<RandomAccessIterator>::difference_type>
void sink(RandomAccessIterator begin, difference_type parentIdx, difference_type n, const Compare& cmp) {
    if (n < 2 || (n - 2) / 2 < parentIdx) {
        return;
    }

    auto parentIter = begin + parentIdx;
    while (2 * parentIdx + 1 < n) {
        auto childIdx = 2 * parentIdx + 1;
        auto childIter = begin + childIdx;

        if (childIdx + 1 < n && cmp(*childIter, *(childIter + 1))) {
            ++childIdx;
            ++childIter;
        }

        if (!cmp(*parentIter, *childIter)) {
            break;
        }

        std::iter_swap(parentIter, childIter);
        parentIdx = childIdx;
        parentIter = childIter;
    }
}

template<typename RandomAccessIterator, typename Compare>
void make_heap(RandomAccessIterator begin, RandomAccessIterator end, const Compare& cmp) {
    auto n = std::distance(begin, end);
    if (n < 2) {
        return;
    }

    for (auto start = (n - 2) / 2; start >= 0; --start) {
        sink(begin, start, n, cmp);
    }
}

template<typename RandomAccessIterator>
inline void make_heap(RandomAccessIterator begin, RandomAccessIterator end) {
    atp::make_heap(std::move(begin), std::move(end), Val_less_val());
}

template<typename RandomAccessIterator, typename Compare>
RandomAccessIterator is_heap_until(RandomAccessIterator begin, RandomAccessIterator end, const Compare& cmp) {
    auto n = std::distance(begin, end);
    auto parentIdx = 0;
    auto childIdx = 1;
    auto parentIter = begin;

    while (childIdx < n) {
        auto childIter = begin + childIdx;
        if (cmp(*parentIter, *childIter)) {
            return childIter;
        }

        ++childIdx;
        ++childIter;
        if (childIdx == n) {
            return end;
        }

        if (cmp(*parentIter, *childIter)) {
            return childIter;
        }

        ++parentIdx;
        ++parentIter;
        childIdx = 2 * parentIdx + 1;
    }
    return end;
}

template<typename RandomAccessIterator>
inline RandomAccessIterator is_heap_until(RandomAccessIterator begin, RandomAccessIterator end) {
    return atp::is_heap_until(std::move(begin), std::move(end), Val_less_val());
}

template<typename RandomAccessIterator, typename Compare>
inline bool is_heap(RandomAccessIterator begin, RandomAccessIterator end, const Compare& cmp) {
    return is_heap_until(begin, end, cmp) == end;
}

template<typename RandomAccessIterator>
inline bool is_heap(RandomAccessIterator begin, RandomAccessIterator end) {
    return atp::is_heap_until(begin, end, Val_less_val()) == end;
}

}// namespace atp

#endif//ATL_HEAP_H
