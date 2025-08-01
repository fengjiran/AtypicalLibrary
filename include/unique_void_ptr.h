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

// A UniqueVoidPtr is an owning smart pointer like unique_ptr, but
// with three major differences:
//
//    1) It is specialized to void
//
//    2) It is specialized for a function pointer deleter
//       void(void* ctx); i.e., the deleter doesn't take a
//       reference to the data, just to a context pointer
//       (erased as void*).  In fact, internally, this pointer
//       is implemented as having an owning reference to
//       context, and a non-owning reference to data; this is why
//       you release_context(), not release() (the conventional
//       API for release() wouldn't give you enough information
//       to properly dispose of the object later.)
//
//    3) The deleter is guaranteed to be called when the unique
//       pointer is destructed and the context is non-null; this is different
//       from std::unique_ptr where the deleter is not called if the
//       data pointer is null.
//
// Some of the methods have slightly different types than std::unique_ptr
// to reflect this.
//

// Note [How UniqueVoidPtr is implemented]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// UniqueVoidPtr solves a common problem for allocators of tensor data, which
// is that the data pointer (e.g., float*) which you are interested in, is not
// the same as the context pointer (e.g., DLManagedTensor) which you need
// to actually deallocate the data.  Under a conventional deleter design, you
// have to store extra context in the deleter itself so that you can actually
// delete the right thing.  Implementing this with standard C++ is somewhat
// error-prone: if you use a std::unique_ptr to manage tensors, the deleter will
// not be called if the data pointer is nullptr, which can cause a leak if the
// context pointer is non-null (and the deleter is responsible for freeing both
// the data pointer and the context pointer).
//
// So, in our reimplementation of unique_ptr, which just store the context
// directly in the unique pointer, and attach the deleter to the context
// pointer itself.  In simple cases, the context pointer is just the pointer
// itself.
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

    NODISCARD void* get() const {
        return data_;
    }

    NODISCARD deleter_type get_deleter() const {
        return ctx_.get_deleter();
    }

    NODISCARD void* get_context() const {
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

        auto x = ctx_.release();
        UNUSED(x);
        ctx_.reset(new_data_and_ctx);
        data_ = new_data_and_ctx;
        return true;
    }

private:
    // Lifetime tied to ctx_
    void* data_;
    std::unique_ptr<void, deleter_type> ctx_;
};

inline bool operator==(const UniqueVoidPtr& sp, std::nullptr_t) noexcept {
    return !sp;
}

inline bool operator==(std::nullptr_t, const UniqueVoidPtr& sp) noexcept {
    return !sp;
}

inline bool operator!=(const UniqueVoidPtr& sp, std::nullptr_t) noexcept {
    return sp;
}

inline bool operator!=(std::nullptr_t, const UniqueVoidPtr& sp) noexcept {
    return sp;
}

}// namespace atp

#endif//UNIQUE_VOID_PTR_H
