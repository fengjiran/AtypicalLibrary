//
// Created by 赵丹 on 24-5-20.
//

#ifndef ATYPICALLIBRARY_EXCEPTION_GUARD_HPP
#define ATYPICALLIBRARY_EXCEPTION_GUARD_HPP

#include "config.hpp"

namespace atp {

template<typename Rollback>
class ExceptionGuard {
public:
    ExceptionGuard() = delete;

    explicit ExceptionGuard(Rollback rollback)
        : rollback_(std::move(rollback)), completed_(false) {}

    // ExceptionGuard(ExceptionGuard&& other) noexcept(std::is_nothrow_constructible_v<Rollback>)
    //     : rollback_(std::move(other.rollback_)), completed_(other.completed_) {
    //     other.completed_ = true;
    // }

    ExceptionGuard(const ExceptionGuard&) = delete;
    ExceptionGuard& operator=(const ExceptionGuard&) = delete;
    ExceptionGuard& operator=(ExceptionGuard&&) = delete;

    void complete() noexcept {
        completed_ = true;
    }

    ~ExceptionGuard() {
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
ExceptionGuard(typename Tag::__allow_ctad...) -> ExceptionGuard<Tag...>;
#endif

// template<typename Rollback>
// using _exception_guard = ExceptionGuard<Rollback>;

template<typename Rollback>
ExceptionGuard<Rollback> make_exception_guard(Rollback rollback) {
    return ExceptionGuard<Rollback>(std::move(rollback));
}

}// namespace atp

#endif//ATYPICALLIBRARY_EXCEPTION_GUARD_HPP
