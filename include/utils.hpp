//
// Created by 赵丹 on 24-5-16.
//

#ifndef ATYPICALLIBRARY_UTILS_HPP
#define ATYPICALLIBRARY_UTILS_HPP

#include <iterator>
#include <type_traits>

namespace atp {

template<typename T>
struct type_identity {
    using type = T;
};

template<typename T>
using type_identity_t = typename type_identity<T>::type;

template<typename...>
using void_t = void;

template<typename T>
T* to_address(T* p) noexcept {
    static_assert(!std::is_function_v<T>, "value is a function type");
    return p;
}

template<typename Iter, typename value_type>
using has_input_iterator_category = std::enable_if<
        std::is_convertible_v<typename std::iterator_traits<Iter>::iterator_category, std::input_iterator_tag> &&
                std::is_constructible_v<value_type, typename std::iterator_traits<Iter>::reference>,
        int>;

template<typename T, T v>
struct integral_constant {
    static constexpr T value = v;
    using value_type = T;
    using type = integral_constant;
    constexpr explicit operator value_type() const noexcept { return value; }
    constexpr value_type operator()() const noexcept { return value; }
};

using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;

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

// propagate on container copy assignment
template<typename alloc, typename = void>
struct _has_propagate_on_container_copy_assignment : false_type {};

template<typename alloc>
struct _has_propagate_on_container_copy_assignment<
        alloc,
        void_t<typename alloc::propagate_on_container_copy_assignment>> : true_type {};

template<typename alloc,
         bool = _has_propagate_on_container_copy_assignment<alloc>::value>
struct propagate_on_container_copy_assignment : false_type {};

template<typename alloc>
struct propagate_on_container_copy_assignment<alloc, true> {
    using type = typename alloc::propagate_on_container_copy_assignment;
};

// propagate on container move assignment
template<typename alloc, typename = void>
struct _has_propagate_on_container_move_assignment : false_type {};

template<typename alloc>
struct _has_propagate_on_container_move_assignment<
        alloc,
        void_t<typename alloc::propagate_on_container_move_assignment>> : true_type {};

template<typename alloc,
         bool = _has_propagate_on_container_move_assignment<alloc>::value>
struct propagate_on_container_move_assignment : false_type {};

template<typename alloc>
struct propagate_on_container_move_assignment<alloc, true> {
    using type = typename alloc::propagate_on_container_move_assignment;
};

// propagate on container swap
template<typename alloc, typename = void>
struct _has_propagate_on_container_swap : false_type {};

template<typename alloc>
struct _has_propagate_on_container_swap<
        alloc,
        void_t<typename alloc::propagate_on_container_swap>> : true_type {};

template<typename alloc,
         bool = _has_propagate_on_container_swap<alloc>::value>
struct propagate_on_container_swap : false_type {};

template<typename alloc>
struct propagate_on_container_swap<alloc, true> {
    using type = typename alloc::propagate_on_container_swap;
};

}// namespace atp

#endif//ATYPICALLIBRARY_UTILS_HPP
