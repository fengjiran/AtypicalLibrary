//
// Created by 赵丹 on 25-7-19.
//

#ifndef FUNCTION_TRAITS_H
#define FUNCTION_TRAITS_H

#include <tuple>
#include <type_traits>

namespace atp {

/**
 * is_function_type<T> is true_type if T is a plain function type (i.e.
 * "Result(Args...)")
 */
template<typename T>
struct is_plain_function_type : std::false_type {};

template<typename R, typename... Args>
struct is_plain_function_type<R(Args...)> : std::true_type {};

template<typename T>
constexpr bool is_plain_function_type_v = is_plain_function_type<T>::value;

template<typename FuncType>
struct function_traits;

// plain function type.
template<typename R, typename... Args>
struct function_traits<R(Args...)> {
    using func_type = R(Args...);
    using return_type = R;
    using args_type_tuple = std::tuple<Args...>;
    static constexpr auto params_num = sizeof...(Args);
};

/**
 * creates a `function_traits` type for a simple function (pointer) or functor (lambda/struct).
 * Currently does not support class methods.
 */
template<typename Functor>
struct infer_function_traits {
    using type = function_traits<std::remove_const_t<decltype(Functor::operator())>>;
};

template<typename R, typename... Args>
struct infer_function_traits<R(Args...)> {
    using type = function_traits<R(Args...)>;
};

template<typename R, typename... Args>
struct infer_function_traits<R(*)(Args...)> {
    using type = function_traits<R(Args...)>;
};

template<typename R, typename... Args>
struct infer_function_traits<std::function<R(Args...)>> {
    using type = function_traits<R(Args...)>;
};

template<typename R, typename... Args>
struct infer_function_traits<std::function<R(*)(Args...)>> {
    using type = function_traits<R(Args...)>;
};

template<typename T>
using infer_function_traits_t = typename infer_function_traits<T>::type;

}// namespace atp

#endif//FUNCTION_TRAITS_H
