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
struct is_function_type : std::false_type {};

template<typename R, typename... Args>
struct is_function_type<R(Args...)> : std::true_type {};

template<typename T>
using is_function_type_t = typename is_function_type<T>::type;

template<typename T>
constexpr bool is_function_type_v = is_function_type<T>::value;

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

// plain function pointer type
template<typename R, typename... Args>
struct function_traits<R (*)(Args...)> : function_traits<R(Args...)> {};

template<typename R, typename... Args>
struct function_traits<std::function<R(Args...)>> : function_traits<R(Args...)> {};

template<typename R, typename... Args>
struct function_traits<std::function<R (*)(Args...)>> : function_traits<R(Args...)> {};

template<typename Functor, typename R, typename... Args>
struct function_traits<R(Args...)> {
    using func_type = std::remove_const_t<decltype(Functor::operator())>;
    using return_type = R;
    using args_type_tuple = std::tuple<Args...>;
    static constexpr auto params_num = sizeof...(Args);
};

// template<typename Functor>
// struct;

}// namespace atp

#endif//FUNCTION_TRAITS_H
