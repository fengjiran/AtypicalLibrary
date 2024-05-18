//
// Created by 赵丹 on 24-5-16.
//

#ifndef OPENXAE_UTILS_HPP
#define OPENXAE_UTILS_HPP

#include <iterator>
#include <type_traits>

namespace XAcceleratorEngine {

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
    static_assert(!std::is_function<T>::value, "value is a function type");
    return p;
}

template<typename Iter, typename value_type>
using has_input_iterator_category = typename std::enable_if<
        std::is_convertible<typename std::iterator_traits<Iter>::iterator_category, std::input_iterator_tag>::value &&
                std::is_constructible<value_type, typename std::iterator_traits<Iter>::reference>::value,
        int>;

template<typename T, T v>
struct integral_constant {
    static constexpr const T value = v;
    using value_type = T;
    using type = integral_constant;
    inline constexpr explicit operator value_type() const noexcept { return value; }
    inline constexpr value_type operator()() const noexcept { return value; }
};

using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;

template<typename alloc, typename = void>
struct _has_select_on_container_copy_construction : false_type {};

template<typename alloc>
struct _has_select_on_container_copy_construction<
        alloc,
        decltype((void) std::declval<alloc>().select_on_container_copy_construction())> : true_type {};

template<typename alloc,
         typename = typename std::enable_if<_has_select_on_container_copy_construction<const alloc>::value>::type>
inline constexpr static alloc select_on_container_copy_construction(const alloc& a) {
    return a.select_on_container_copy_construction();
}

template<typename alloc,
         typename = void,
         typename = typename std::enable_if<!_has_select_on_container_copy_construction<const alloc>::value>::type>
inline constexpr static alloc select_on_container_copy_construction(const alloc& a) {
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

}// namespace XAcceleratorEngine

#endif//OPENXAE_UTILS_HPP
