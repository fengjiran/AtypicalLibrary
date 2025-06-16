//
// Created by 赵丹 on 25-6-12.
//

#ifndef TENSOR_H
#define TENSOR_H

#include <cstdint>
#include <vector>

#ifdef __has_cpp_attribute
#if __has_cpp_attribute(nodiscard)
#define NODISCARD [[nodiscard]]
#else
#define NODISCARD
#endif

#if __has_cpp_attribute(maybe_unused)
#define MAYBE_UNUSED [[maybe_unused]]
#else
#define MAYBE_UNUSED
#endif
#endif

#define UNUSED(expr)   \
    do {               \
        (void) (expr); \
    } while (false)


inline int32_t atomic_inc_relaxed(int32_t* ptr) {
    return __atomic_fetch_add(ptr, 1, __ATOMIC_RELAXED);
}

inline int32_t atomic_dec_rel_acq(int32_t* ptr) {
    return __atomic_fetch_sub(ptr, 1, __ATOMIC_ACQ_REL);
}

inline int32_t atomic_load_relaxed(const int32_t* ptr) {
    auto raw_ptr = const_cast<int32_t*>(ptr);
    return __atomic_load_n(raw_ptr, __ATOMIC_RELAXED);
}

enum class DeviceType : uint8_t {
    kCPU = 0,
    kCUDA = 1,
};

enum class DLDataTypeCode : uint8_t {
    kInt = 0,
    kUInt = 1,
    kFloat = 2,
};

struct Device {
    DeviceType device_type;
    uint8_t device_id;
};

struct DLDataType {
    // Data type code.
    DLDataTypeCode code;

    // Number of bits per data element.
    // For example, 8 for int8_t, 32 for float.
    int8_t bits;

    // Number of lanes per data element, for vector.
    int16_t lanes;
};

struct TensorInfo {
    void* data;
    int32_t ndim;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    DLDataType dtype;
    Device device;
    int32_t ref_counter;
};


template<typename Allocator>
class TensorNode {
public:
    TensorNode() : data(nullptr), ndim(0), dtype({DLDataTypeCode::kFloat, 32, 1}),
                   device(DeviceType::kCPU), ref_counter(0) {}

    TensorNode(Allocator alloc, std::vector<int64_t> shape, DLDataType dtype, DeviceType device)
        : shape(std::move(shape)) {}

    NODISCARD int32_t use_count() const {
        return atomic_load_relaxed(&ref_counter);
    }

    NODISCARD bool unique() const {
        return use_count() == 1;
    }

private:
    void inc_ref() {
        atomic_inc_relaxed(&ref_counter);
    }

    void* data;
    int32_t ndim;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    DLDataType dtype;
    Device device;
    Allocator alloc;

    int32_t ref_counter;
};


#endif//TENSOR_H
