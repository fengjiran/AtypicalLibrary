//
// Created by 赵丹 on 24-5-20.
//

#ifndef ATYPICALLIBRARY_EXCEPTION_GUARD_HPP
#define ATYPICALLIBRARY_EXCEPTION_GUARD_HPP

#include "config.hpp"

namespace atp {

template<typename Rollback>
class _exception_guard_exceptions {
public:
    _exception_guard_exceptions() = delete;

    explicit _exception_guard_exceptions(Rollback rollback)
        : rollback_(std::move(rollback)), complete_(false) {}

    //TODO: move constructor

private:
    Rollback rollback_;
    bool complete_;
};

}// namespace atp

#endif//ATYPICALLIBRARY_EXCEPTION_GUARD_HPP
