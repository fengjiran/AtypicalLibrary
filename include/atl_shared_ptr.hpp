//
// Created by richard on 8/28/24.
//

#ifndef ATYPICALLIBRARY_ATL_SHARED_PTR_HPP
#define ATYPICALLIBRARY_ATL_SHARED_PTR_HPP

#include "utils.hpp"

namespace atp {

template<typename T>
class shared_ptr {
public:
    shared_ptr() : p(nullptr), useCnt(nullptr) {}

    explicit shared_ptr(T* pt) : p(pt), useCnt(new size_t(1)) {}

    shared_ptr(const shared_ptr& sp) : p(sp.p), useCnt(sp.useCnt) {
        if (useCnt) {
            ++*useCnt;
        }
    }

    shared_ptr(shared_ptr&& sp) noexcept : p(sp.p), useCnt(sp.useCnt) {
        sp.p = nullptr;
        sp.useCnt = nullptr;
    }

    shared_ptr& operator=(const shared_ptr& rhs) {
        //        if (this != &rhs) {
        //            if (rhs.useCnt) {
        //                ++*rhs.useCnt;
        //            }
        //
        //            if (useCnt && --*useCnt == 0) {
        //                delete p;
        //                delete useCnt;
        //            }
        //
        //            p = rhs.p;
        //            useCnt = rhs.useCnt;
        //        }

        // use copy and swap
        shared_ptr tmp(rhs);
        swap(tmp, *this);
        return *this;
    }

    shared_ptr& operator=(shared_ptr&& rhs) noexcept {
        // use copy and swap
        shared_ptr tmp(rhs);
        swap(tmp, *this);
        return *this;
    }


    friend void swap(shared_ptr& a, shared_ptr& b) noexcept {
        std::swap(a.p, b.p);
        std::swap(a.useCnt, b.useCnt);
    }

    ~shared_ptr() {
        if (useCnt && --*useCnt == 0) {
            delete p;
            delete useCnt;
        }
    }

private:
    T* p;
    size_t* useCnt;
};

}// namespace atp

#endif//ATYPICALLIBRARY_ATL_SHARED_PTR_HPP
