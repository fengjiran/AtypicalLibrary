//
// Created by 赵丹 on 25-4-16.
//

#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <vector>

template<typename R, typename... Args>
class Functor {
    using function_type = std::function<R(Args...)>;
    std::vector<function_type> func_;

public:

};

#endif//THREAD_POOL_H
