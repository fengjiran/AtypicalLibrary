//
// Created by richard on 6/24/25.
//

#ifndef STORAGE_IMPL_H
#define STORAGE_IMPL_H

#include "tensor_utils.h"
#include "unique_void_ptr.h"

namespace atp {

// A DataPtr is a unique pointer (with an attached deleter and some
// context for the deleter) to some memory, which also records what
// device is for its data.
//
// nullptr DataPtrs can still have a nontrivial device; this allows
// us to treat zero-size allocations uniformly with non-zero allocations.
//
class DataPtr {
public:
    DataPtr() : device_(DeviceType::kCPU) {}

    DataPtr(void* data, DeviceType device) : ptr_(data), device_(device) {}

    DataPtr(void* data, void* ctx, deleter_type deleter, DeviceType device)
        : ptr_(data, ctx, deleter), device_(device) {}

    NODISCARD DeviceType device() const {
        return device_;
    }

    void clear() {
        ptr_.clear();
    }

    NODISCARD void* get() const {
        return ptr_.get();
    }

    NODISCARD void* get_context() const {
        return ptr_.get_context();
    }

    NODISCARD deleter_type get_deleter() const {
        return ptr_.get_deleter();
    }

    void* release_context() {
        return ptr_.release_context();
    }

    std::unique_ptr<void, deleter_type>&& move_context() {
        return ptr_.move_context();
    }

    operator bool() const {
        return ptr_;
    }

    void* operator->() const {
        return ptr_.get();
    }

    bool compare_exchange_deleter(deleter_type expected_deleter, deleter_type new_deleter) {
        return ptr_.compare_exchange_deleter(expected_deleter, new_deleter);
    }

    template<typename T>
    T* cast_context(deleter_type expected_deleter) const {
        return ptr_.cast_context<T>(expected_deleter);
    }

    bool unsafe_reset_data_and_ctx(void* new_data_and_ctx) {
        return ptr_.unsafe_reset_data_and_ctx(new_data_and_ctx);
    }

    void unsafe_set_device(DeviceType device) {
        device_ = device;
    }

private:
    UniqueVoidPtr ptr_;
    DeviceType device_;
};

inline bool operator==(const DataPtr& dp, std::nullptr_t) noexcept {
    return !dp;
}

inline bool operator==(std::nullptr_t, const DataPtr& dp) noexcept {
    return !dp;
}

inline bool operator!=(const DataPtr& dp, std::nullptr_t) noexcept {
    return dp;
}

inline bool operator!=(std::nullptr_t, const DataPtr& dp) noexcept {
    return dp;
}

}

#endif //STORAGE_IMPL_H
