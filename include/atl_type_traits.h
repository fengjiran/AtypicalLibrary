//
// Created by richard on 10/25/24.
//

#ifndef ATL_TYPE_TRAITS_H
#define ATL_TYPE_TRAITS_H

namespace atp {

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

template<bool b>
using bool_constant = integral_constant<bool, b>;

template<typename...>
using void_t = void;

template<typename T>
struct type_identity {
    using type = T;
};

template<typename T>
using type_identity_t = typename type_identity<T>::type;

// remove reference
template<typename T>
struct remove_reference {
    using type = T;
};

template<typename T>
struct remove_reference<T&> {
    using type = T;
};

template<typename T>
struct remove_reference<T&&> {
    using type = T;
};

template<typename T>
using remove_reference_t = typename remove_reference<T>::type;

// is_same_v
template<typename T, typename U>
constexpr bool is_same_v = false;

template<typename T>
constexpr bool is_same_v<T, T> = true;

// template<typename T, typename U>
// struct is_same : false_type{};
//
// template<typename T, typename U = T>
// struct is_same<T, U> : true_type{};
//
// template<typename T, typename U>
// constexpr bool is_same_v = is_same<T, U>::value;

}// namespace atp

#endif//ATL_TYPE_TRAITS_H
