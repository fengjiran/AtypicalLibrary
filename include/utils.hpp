//
// Created by 赵丹 on 24-5-16.
//

#ifndef ATYPICALLIBRARY_UTILS_HPP
#define ATYPICALLIBRARY_UTILS_HPP

#include "atl_type_traits.h"
#include <chrono>
#include <iterator>
#include <type_traits>

namespace atp {

class Timer {
public:
    Timer() : start(std::chrono::high_resolution_clock::now()) {}

    // Get elapsed time in seconds
    [[nodiscard]] double GetElapsedTime() const {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return static_cast<double>(duration.count()) *
               std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
    }

private:
    std::chrono::high_resolution_clock::time_point start;
};

// template<typename T>
// T* to_address(T* p) noexcept {
//     static_assert(!std::is_function_v<T>, "value is a function type");
//     return p;
// }

template<typename Iter, typename value_type>
using has_input_iterator_category = std::enable_if<
        std::is_convertible_v<typename std::iterator_traits<Iter>::iterator_category, std::input_iterator_tag> &&
                std::is_constructible_v<value_type, typename std::iterator_traits<Iter>::reference>,
        int>;


#define HAS_PROPERTY(NAME, PROPERTY)      \
    template<typename T, typename = void> \
    struct NAME : false_type {};          \
    template<typename T>                  \
    struct NAME<T, void_t<typename T::PROPERTY>> : true_type {}

// propagate on container copy assignment
// HAS_PROPERTY(has_propagate_on_container_copy_assignment, propagate_on_container_copy_assignment);

// propagate on container move assignment
// HAS_PROPERTY(has_propagate_on_container_move_assignment, propagate_on_container_move_assignment);

// propagate on container swap
// HAS_PROPERTY(has_propagate_on_container_swap, propagate_on_container_swap);

template<typename, typename = void>
struct _has_select_on_container_copy_construction : false_type {};

template<typename alloc>
struct _has_select_on_container_copy_construction<
        alloc,
        decltype((void) std::declval<alloc>().select_on_container_copy_construction())> : true_type {};

template<typename alloc,
         typename = std::enable_if_t<_has_select_on_container_copy_construction<const alloc>::value>>
constexpr static alloc select_on_container_copy_construction(const alloc& a) {
    return a.select_on_container_copy_construction();
}

template<typename alloc,
         typename = void,
         typename = std::enable_if_t<!_has_select_on_container_copy_construction<const alloc>::value>>
constexpr static alloc select_on_container_copy_construction(const alloc& a) {
    return a;
}

// template<typename alloc,
//          bool = has_propagate_on_container_copy_assignment<alloc>::value>
// struct propagate_on_container_copy_assignment : false_type {};
//
// template<typename alloc>
// struct propagate_on_container_copy_assignment<alloc, true> {
//     using type = typename alloc::propagate_on_container_copy_assignment;
// };

// template<typename alloc,
//          bool = has_propagate_on_container_move_assignment<alloc>::value>
// struct propagate_on_container_move_assignment : false_type {};
//
// template<typename alloc>
// struct propagate_on_container_move_assignment<alloc, true> {
//     using type = typename alloc::propagate_on_container_move_assignment;
// };

// template<typename alloc,
//          bool = has_propagate_on_container_swap<alloc>::value>
// struct propagate_on_container_swap : false_type {};
//
// template<typename alloc>
// struct propagate_on_container_swap<alloc, true> {
//     using type = typename alloc::propagate_on_container_swap;
// };

}// namespace atp

#endif//ATYPICALLIBRARY_UTILS_HPP
