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
        : rollback_(std::move(rollback)), completed_(false) {}

    _exception_guard_exceptions(_exception_guard_exceptions&& other) noexcept(std::is_nothrow_constructible<Rollback>::value)
        : rollback_(std::move(other.rollback_)), completed_(other.completed_) {
        other.completed_ = true;
    }

    _exception_guard_exceptions(const _exception_guard_exceptions&) = delete;
    _exception_guard_exceptions& operator=(const _exception_guard_exceptions&) = delete;
    _exception_guard_exceptions& operator=(_exception_guard_exceptions&&) = delete;

    void complete() noexcept {
        completed_ = true;
    }

    ~_exception_guard_exceptions() {
        if (!completed_) {
            rollback_();
        }
    }

private:
    Rollback rollback_;
    bool completed_;
};

//std::is_nothrow_constructible

}// namespace atp

#endif//ATYPICALLIBRARY_EXCEPTION_GUARD_HPP
