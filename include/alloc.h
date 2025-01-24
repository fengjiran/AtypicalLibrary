//
// Created by 赵丹 on 25-1-23.
//

#ifndef ALLOC_H
#define ALLOC_H

#include "base.h"

namespace base {

enum class MemcpyKind {
    kMemcpyCPU2CPU = 0,
    kMemcpyCPU2CUDA = 1,
    kMemcpyCUDA2CPU = 2,
    kMemcpyCUDA2CUDA = 3,
};

class DeviceAllocator {
public:
    explicit DeviceAllocator(DeviceType device_type) : device_type_(device_type) {}

    virtual ~DeviceAllocator() = default;

    virtual DeviceType device_type() const { return device_type_; }

    virtual void release(void* ptr) const = 0;

    virtual void* allocate(size_t byte_size) const = 0;

    virtual void memcpy(const void* src_ptr, void* dest_ptr,
                        size_t byte_size,
                        MemcpyKind memcpy_kind = MemcpyKind::kMemcpyCPU2CPU,
                        void* stream = nullptr,
                        bool need_sync = false) const;

    virtual void memset_zero(void* ptr, size_t byte_size, void* stream, bool need_sync = false);

private:
    DeviceType device_type_{DeviceType::kDeviceUnknown};
};

}// namespace base

#endif//ALLOC_H
