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
void swim(RandomAccessIterator begin, difference_type childIdx, difference_type n, const Compare& cmp) {
    if (n < 2) {
        return;
    }

    auto childIter = begin + childIdx;
    auto bottomVal = std::move(*childIter);
    auto parentIdx = (childIdx - 1) / 2;
    auto parentIter = begin + parentIdx;

    while (childIdx > 1 && cmp(*parentIter, bottomVal)) {
        *childIter = std::move(*parentIter);
        childIdx = parentIdx;
        childIter = parentIter;

        parentIdx = (childIdx - 1) / 2;
        parentIter = begin + parentIdx;
    }
    *parentIter = std::move(bottomVal);
}

template<typename RandomAccessIterator,
         typename Compare,
         typename difference_type = typename std::iterator_traits<RandomAccessIterator>::difference_type>
void sink(RandomAccessIterator begin, difference_type parentIdx, difference_type n, const Compare& cmp) {
    if (n < 2 || parentIdx > (n - 2) / 2) {
        return;
    }

    auto parentIter = begin + parentIdx;
    auto topVal = std::move(*parentIter);
    while (2 * parentIdx + 1 < n) {
        auto childIdx = 2 * parentIdx + 1;
        auto childIter = begin + childIdx;

        if (childIdx + 1 < n && cmp(*childIter, *(childIter + 1))) {
            ++childIdx;
            ++childIter;
        }

        if (!cmp(topVal, *childIter)) {
            break;
        }

        // std::iter_swap(parentIter, childIter);
        *parentIter = std::move(*childIter);
        parentIdx = childIdx;
        parentIter = childIter;
    }
    *parentIter = std::move(topVal);
}

template<typename RandomAccessIterator, typename Compare>
void push_heap(RandomAccessIterator begin, RandomAccessIterator end, const Compare& cmp) {
    swim(begin, end - begin - 1, end - begin, cmp);
}

template<typename RandomAccessIterator>
void push_heap(RandomAccessIterator begin, RandomAccessIterator end) {
    swim(begin, end - begin - 1, end - begin, Val_less_val());
}

template<typename RandomAccessIterator,
typename Compare,
typename difference_type = typename std::iterator_traits<RandomAccessIterator>::difference_type>
void pop_heap(RandomAccessIterator begin, RandomAccessIterator end, const Compare& cmp) {
    if (end - begin > 1) {
        --end;
        std::iter_swap(begin, end);
        // *begin = std::move(*end);
        sink(begin, static_cast<difference_type>(0), static_cast<difference_type>(end - begin), cmp);
    }
}

template<typename RandomAccessIterator>
void pop_heap(RandomAccessIterator begin, RandomAccessIterator end) {
    atp::pop_heap(begin, end, Val_less_val());
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
void make_heap(RandomAccessIterator begin, RandomAccessIterator end) {
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
RandomAccessIterator is_heap_until(RandomAccessIterator begin, RandomAccessIterator end) {
    return atp::is_heap_until(std::move(begin), std::move(end), Val_less_val());
}

template<typename RandomAccessIterator, typename Compare>
bool is_heap(RandomAccessIterator begin, RandomAccessIterator end, const Compare& cmp) {
    return is_heap_until(begin, end, cmp) == end;
}

template<typename RandomAccessIterator>
bool is_heap(RandomAccessIterator begin, RandomAccessIterator end) {
    return atp::is_heap_until(begin, end, Val_less_val()) == end;
}

}// namespace atp

#endif//ATL_HEAP_H
