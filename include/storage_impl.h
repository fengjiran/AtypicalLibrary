//
// Created by richard on 6/24/25.
//

#ifndef STORAGE_IMPL_H
#define STORAGE_IMPL_H

#include "allocator.h"

namespace atp {

class StorageImpl {
public:
    StorageImpl(size_t nbytes, DataPtr data_ptr, const std::unique_ptr<Allocator>& alloc)
        : nbytes_(nbytes), data_ptr_(std::move(data_ptr)), alloc_(alloc) {}

    StorageImpl(size_t nbytes, const std::unique_ptr<Allocator>& alloc)
        : nbytes_(nbytes), data_ptr_(alloc->allocate(nbytes)), alloc_(alloc) {}

    NODISCARD size_t nbytes() const {
        return nbytes_;
    }

    NODISCARD DataPtr& data_ptr() {
        return data_ptr_;
    }

    NODISCARD const DataPtr& const_data_ptr() const {
        return data_ptr_;
    }

    NODISCARD void* get() const {
        return data_ptr_.get();
    }

    NODISCARD const void* const_get() const {
        return data_ptr_.get();
    }

    NODISCARD DeviceType device() const {
        return data_ptr_.device();
    }

    StorageImpl() = delete;
    StorageImpl(const StorageImpl&) = delete;
    StorageImpl(StorageImpl&&) noexcept = delete;
    StorageImpl& operator=(const StorageImpl&) = delete;
    StorageImpl& operator=(StorageImpl&&) noexcept = delete;

private:
    size_t nbytes_;
    DataPtr data_ptr_;
    const std::unique_ptr<Allocator>& alloc_;
};

}// namespace atp

#endif//STORAGE_IMPL_H
