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

// remove const
template<typename T>
struct remove_const {
    using type = T;
};

template<typename T>
struct remove_const<const T> {
    using type = T;
};

template<typename T>
using remove_const_t = typename remove_const<T>::type;

// remove volatile
template<typename T>
struct remove_volatile {
    using type = T;
};

template<typename T>
struct remove_volatile<volatile T> {
    using type = T;
};

template<typename T>
using remove_volatile_t = typename remove_volatile<T>::type;

// remove cv
template<typename T>
struct remove_cv {
    using type = remove_volatile_t<remove_const_t<T>>;
};

template<typename T>
using remove_cv_t = remove_volatile_t<remove_const_t<T>>;

// is_void
template<typename T>
struct is_void : false_type {};

template<>
struct is_void<void> : true_type {};

template<>
struct is_void<const void> : true_type {};

template<>
struct is_void<volatile void> : true_type {};

template<>
struct is_void<const volatile void> : true_type {};

template<typename T>
constexpr bool is_void_v = is_void<T>::value;

// if
template<bool>
struct IfImpl;

template<>
struct IfImpl<true> {
    template<typename IfRes, typename ElseRes>
    using select = IfRes;
};

template<>
struct IfImpl<false> {
    template<typename IfRes, typename ElseRes>
    using select = ElseRes;
};

template<bool cond, typename IfRes, typename ElseRes>
using If = typename IfImpl<cond>::template select<IfRes, ElseRes>;

// conditional
// template<bool b, typename T1, typename T2>
// struct conditional {};
//
// template<typename T1, typename T2>
// struct conditional<true, T1, T2> {
//     using type = T1;
// };
//
// template<typename T1, typename T2>
// struct conditional<false, T1, T2> {
//     using type = T2;
// };
//
// template<bool b, typename T1, typename T2>
// using conditional_t = typename conditional<b, T1, T2>::type;


template<bool b, typename IfRes, typename ElseRes>
struct conditional {
    using type = IfRes;
};

template<typename IfRes, typename ElseRes>
struct conditional<false, IfRes, ElseRes> {
    using type = ElseRes;
};

template<bool b, typename IfRes, typename ElseRes>
using conditional_t = typename conditional<b, IfRes, ElseRes>::type;


}// namespace atp

#endif//ATL_TYPE_TRAITS_H
