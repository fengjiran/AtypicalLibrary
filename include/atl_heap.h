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
void sink(RandomAccessIterator begin, RandomAccessIterator target, difference_type n, const Compare& cmp) {
    difference_type parentIdx = target - begin;
    auto parentIter = begin + parentIdx;

    if (n < 2 || (n - 2) / 2 < parentIdx) {
        return;
    }

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
        sink(begin, begin + start, n, cmp);
    }
}

template<typename RandomAccessIterator>
void make_heap(RandomAccessIterator begin, RandomAccessIterator end) {
    make_heap(begin, end, Val_less_val());
}



}// namespace atp

#endif//ATL_HEAP_H
