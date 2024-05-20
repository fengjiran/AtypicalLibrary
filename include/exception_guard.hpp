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

#if CPP_STD_VERSION >= 17
template<typename... Tag>
_exception_guard_exceptions(typename Tag::__allow_ctad...)
        -> _exception_guard_exceptions<Tag...>;
#endif

template<typename Rollback>
using _exception_guard = _exception_guard_exceptions<Rollback>;

template<typename Rollback>
_exception_guard<Rollback> _make_exception_guard(Rollback rollback) {
    return _exception_guard<Rollback>(std::move(rollback));
}

}// namespace atp

#endif//ATYPICALLIBRARY_EXCEPTION_GUARD_HPP
