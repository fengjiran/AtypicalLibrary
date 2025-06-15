//
// Created by 赵丹 on 25-6-12.
//

#ifndef TENSOR_H
#define TENSOR_H

#include <cstdint>
#include <vector>

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

class TensorNode {
public:

private:
    void inc_ref() {
        atomic_inc_relaxed(&ref_counter);
    }

    void* data;
    int32_t ndim;
    int64_t* shape;
    int64_t* strides;
    DLDataType dtype;
    Device device;

    int32_t ref_counter;
};



#endif//TENSOR_H
