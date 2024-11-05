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

template<typename T, typename U>
struct has_rebind {
private:
    template<typename X>
    static false_type test(...);

    template<typename X>
    static true_type test(typename X::template rebind<U>* = nullptr);

public:
    static const bool value = decltype(test<T>(nullptr))::value;
};

template<typename T, typename U, bool = has_rebind<T, U>::value>
struct pointer_traits_rebind {
    using type = typename T::template rebind<U>;
};

template<template<typename, typename...> typename Sp, typename T, typename... Args, typename U>
struct pointer_traits_rebind<Sp<T, Args...>, U, true> {
    using type = typename Sp<T, Args...>::template rebind<U>;
};

template<template<typename, typename...> typename Sp, typename T, typename... Args, typename U>
struct pointer_traits_rebind<Sp<T, Args...>, U, false> {
    using type = Sp<T, Args...>;
};

template<typename Ptr, typename = void>
struct pointer_traits_impl {};

template<typename Ptr>
struct pointer_traits_impl<Ptr, void_t<typename pointer_traits_element_type<Ptr>::type>> {
    using pointer = Ptr;
    using element_type = typename pointer_traits_element_type<pointer>::type;
    using difference_type = typename pointer_traits_difference_type<pointer>::type;

    template<typename U>
    using rebind = typename pointer_traits_rebind<pointer, U>::type;

private:
    struct nat {};

public:
    static pointer pointer_to(conditional_t<is_void_v<element_type>, nat, element_type>& r) {
        return pointer::pointer_to(r);
    }
};

template<typename Ptr>
struct pointer_traits : pointer_traits_impl<Ptr> {};

template<typename T>
struct pointer_traits<T*> {
    using pointer = T*;
    using element_type = T;
    using difference_type = std::ptrdiff_t;

    template<typename U>
    using rebind = U*;

private:
    struct nat {};

public:
    static pointer pointer_to(conditional_t<is_void_v<element_type>, nat, element_type>& r) {
        return std::addressof(r);
    }
};

template<typename From, typename To>
using rebind_pointer_t = typename pointer_traits<From>::template rebind<To>;


template<typename T>
constexpr T* to_address(T* p) noexcept {
    static_assert(!is_function_v<T>, "value is a function type");
    return p;
}

template<typename Pointer, typename = void>
struct HasToAddress : false_type {};

template<typename Pointer>
struct HasToAddress<Pointer,
                    decltype((void) pointer_traits<Pointer>::to_address(std::declval<const Pointer&>()))> : true_type {};


template<typename Pointer, typename = void>
struct HasArrow : false_type {};

template<typename Pointer>
struct HasArrow<Pointer, decltype((void) std::declval<const Pointer&>().operator->())> : true_type {};

template<typename Pointer>
struct IsFancyPointer {
    static const bool value = HasArrow<Pointer>::value || HasToAddress<Pointer>::value;
};

}// namespace atp

#endif//ATL_POINTER_TRAITS_H
