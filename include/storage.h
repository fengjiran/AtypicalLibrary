//
// Created by 赵丹 on 25-6-27.
//

#ifndef STORAGE_H
#define STORAGE_H

#include "storage_impl.h"

namespace atp {

class Storage {
public:
    Storage() = default;

    Storage(std::shared_ptr<StorageImpl> ptr)
        : impl_(std::move(ptr)) {}

    // Allocates memory buffer using the given allocator and creates a storage with it
    Storage(size_t nbytes, const std::unique_ptr<Allocator>& alloc)
        : impl_(std::make_shared<StorageImpl>(nbytes, alloc)) {}

    // Creates storage with pre-allocated memory buffer.
    Storage(size_t nbytes, DataPtr data_ptr, const std::unique_ptr<Allocator>& alloc)
        : impl_(std::make_shared<StorageImpl>(nbytes, std::move(data_ptr), alloc)) {}

    NODISCARD size_t nbytes() const {
        return impl_ ? impl_->nbytes() : 0;
    }

    NODISCARD void* data() const {
        return impl_ ? impl_->get() : nullptr;
    }

    NODISCARD const void* const_data() const {
        return impl_ ? impl_->get() : nullptr;
    }

    NODISCARD DataPtr& data_ptr() const {
        CHECK(impl_ != nullptr) << "Storage is not initialized";
        return impl_->data_ptr();
    }

    NODISCARD const DataPtr& const_data_ptr() const {
        CHECK(impl_ != nullptr) << "Storage is not initialized";
        return impl_->data_ptr();
    }

    NODISCARD DeviceType device() const {
        CHECK(impl_ != nullptr) << "Storage is not initialized";
        return impl_->device();
    }

    operator bool() const {
        return impl_ != nullptr;
    }

    NODISCARD size_t use_count() const {
        return impl_.use_count();
    }

    NODISCARD bool unique() const {
        return impl_.use_count() == 1;
    }


private:
    std::shared_ptr<StorageImpl> impl_;
};

// Packed container for TensorImpl shape and strides.
// 1 size_t for the size
// 5 eightbytes of inline sizes and 5 eightbytes of inline strides, OR pointer
// to out-of-line array
template<size_t MAX_INLINE_SIZE = 5>
class ShapeAndStride {
public:
    NODISCARD size_t size() const noexcept {
        return size_;
    }

    NODISCARD int64_t* shape_data() noexcept {
        return is_inline() ? &inline_storage_[0] : &out_of_line_storage_[0];
    }

    NODISCARD const int64_t* shape_data() const noexcept {
        return is_inline() ? &inline_storage_[0] : &out_of_line_storage_[0];
    }

    NODISCARD int64_t* stride_data() noexcept {
        return is_inline() ? &inline_storage_[MAX_INLINE_SIZE] : &out_of_line_storage_[size()];
    }

    NODISCARD const int64_t* stride_data() const noexcept {
        return is_inline() ? &inline_storage_[MAX_INLINE_SIZE] : &out_of_line_storage_[size()];
    }

private:
    NODISCARD bool is_inline() const noexcept {
        return size_ <= MAX_INLINE_SIZE;
    }

    size_t size_{1};
    union {
        int64_t* out_of_line_storage_;
        int64_t inline_storage_[MAX_INLINE_SIZE * 2];
    };
};

}// namespace atp

#endif//STORAGE_H
