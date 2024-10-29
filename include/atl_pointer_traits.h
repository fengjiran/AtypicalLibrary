//
// Created by richard on 10/29/24.
//

#ifndef ATL_POINTER_TRAITS_H
#define ATL_POINTER_TRAITS_H

#include "atl_type_traits.h"

namespace atp {

template<typename T, typename = void>
struct has_element_type : false_type {};

template<typename T>
struct has_element_type<T, void_t<typename T::element_type>> : true_type {};

template<typename Ptr, bool = has_element_type<Ptr>::value>
struct pointer_traits_element_type {};

template<typename Ptr>
struct pointer_traits_element_type<Ptr, true> {
    using type = typename Ptr::element_type;
};

template<template<typename, typename...> typename Sp, typename T, typename... Args>
struct pointer_traits_element_type<Sp<T, Args...>, true> {
    using type = typename Sp<T, Args...>::element_type;
};

template<template<typename, typename...> typename Sp, typename T, typename... Args>
struct pointer_traits_element_type<Sp<T, Args...>, false> {
    using type = T;
};

template<typename T, typename = void>
struct has_difference_type : false_type {};

template<typename T>
struct has_difference_type<T, void_t<typename T::difference_type>> : true_type {};

template<typename Ptr, bool = has_difference_type<Ptr>::value>
struct pointer_traits_difference_type {
    using type = std::ptrdiff_t;
};

template<typename Ptr>
struct pointer_traits_difference_type<Ptr, true> {
    using type = typename Ptr::difference_type;
};

}// namespace atp

#endif//ATL_POINTER_TRAITS_H
