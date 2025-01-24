//
// Created by 赵丹 on 25-1-23.
//

#ifndef ATL_DISTANCE_H
#define ATL_DISTANCE_H

#include <iterator>
#include <type_traits>

namespace atp {

// using SFINAE
template<typename Iter>
constexpr bool is_random_access_iter =
        std::is_same_v<typename std::iterator_traits<Iter>::iterator_category, std::random_access_iterator_tag>;

template<typename Iter,
         typename = std::enable_if_t<is_random_access_iter<Iter>>>
typename std::iterator_traits<Iter>::difference_type
distance(Iter first, Iter last) {
    return last - first;
}

template<typename Iter,
         typename = void, // avoid template parameter redefines default argument
         typename = std::enable_if_t<!is_random_access_iter<Iter>>>
typename std::iterator_traits<Iter>::difference_type
distance(Iter first, Iter last) {
    typename std::iterator_traits<Iter>::difference_type dist = 0;
    while (first != last) {
        ++first;
        ++dist;
    }
    return dist;
}

// function overload
template<typename Iter>
typename std::iterator_traits<Iter>::difference_type
_distance(Iter first, Iter last, std::random_access_iterator_tag) {
    return last - first;
}

template<typename Iter>
typename std::iterator_traits<Iter>::difference_type
_distance(Iter first, Iter last, std::input_iterator_tag) {
    typename std::iterator_traits<Iter>::difference_type dist = 0;
    while (first != last) {
        ++first;
        ++dist;
    }
    return dist;
}

// template<typename Iter>
// typename std::iterator_traits<Iter>::difference_type
// distance(Iter first, Iter last) {
//     using iterator_tag = typename std::iterator_traits<Iter>::iterator_category;
//     return _distance(first, last, iterator_tag());
// }

// if constexpr
// template<typename Iter>
// typename std::iterator_traits<Iter>::difference_type
// distance(Iter first, Iter last) {
//     using iterator_tag = typename std::iterator_traits<Iter>::iterator_category;
//     typename std::iterator_traits<Iter>::difference_type dist = 0;
//
//     if constexpr (std::is_same_v<iterator_tag, std::random_access_iterator_tag>) {
//         dist = last - first;
//     } else {
//         while (first != last) {
//             ++first;
//             ++dist;
//         }
//     }
//     return dist;
// }

}// namespace atp

#endif//ATL_DISTANCE_H
