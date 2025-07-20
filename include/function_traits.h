//
// Created by 赵丹 on 25-7-19.
//

#ifndef FUNCTION_TRAITS_H
#define FUNCTION_TRAITS_H

#include <tuple>
#include <type_traits>

namespace atp {

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

template<typename R, typename... Args>
struct function_traits<R (*)(Args...)> : function_traits<R(Args...)> {};

template<typename R, typename... Args>
struct function_traits<std::function<R(Args...)>> : function_traits<R(Args...)> {};

template<typename R, typename... Args>
struct function_traits<std::function<R (*)(Args...)>> : function_traits<R(Args...)> {};

}// namespace atp

#endif//FUNCTION_TRAITS_H
