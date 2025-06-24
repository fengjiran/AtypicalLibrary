//
// Created by 赵丹 on 25-6-24.
//

#ifndef UNIQUE_VOID_PTR_H
#define UNIQUE_VOID_PTR_H

#include <memory>

namespace atp {

using deleter_type = void (*)(void*);

inline void delete_nothing(void* ptr) {
    (void) ptr;
}

class UniqueVoidPtr {
public:
    UniqueVoidPtr() : data_(nullptr), ctx_(nullptr, delete_nothing) {}

    explicit UniqueVoidPtr(void* ptr) : data_(ptr), ctx_(nullptr, delete_nothing) {}

    UniqueVoidPtr(void* ptr, void* ctx, deleter_type deleter)
        : data_(ptr), ctx_(ctx, deleter ? deleter : delete_nothing) {}

    void clear() {
        data_ = nullptr;
        ctx_ = nullptr;
    }

    void* get() const {
        return data_;
    }

    deleter_type get_deleter() const {
        return ctx_.get_deleter();
    }

    void* get_context() const {
        return ctx_.get();
    }

    void* release_context() {
        return ctx_.release();
    }

    void* operator->() const {
        return data_;
    }

    operator bool() const {
        return data_ || ctx_;
    }

    std::unique_ptr<void, deleter_type>&& move_context() {
        return std::move(ctx_);
    }

    bool compare_exchange_deleter(deleter_type expected_deleter, deleter_type new_deleter) {
        if (get_deleter() != expected_deleter) {
            return false;
        }
        ctx_ = std::unique_ptr<void, deleter_type>(ctx_.release(), new_deleter);
        return true;
    }

    template<typename T>
    T* cast_context(deleter_type expected_deleter) const {
        if (get_deleter() != expected_deleter) {
            return nullptr;
        }

        return static_cast<T*>(get_context());
    }

    bool unsafe_reset_data_and_ctx(void* new_data_and_ctx) {
        if (get_deleter() != delete_nothing) {
            return false;
        }

        ctx_.release();
        ctx_.reset(new_data_and_ctx);
        data_ = new_data_and_ctx;
        return true;
    }

private:
    void* data_;
    std::unique_ptr<void, deleter_type> ctx_;
};

}// namespace atp

#endif//UNIQUE_VOID_PTR_H
