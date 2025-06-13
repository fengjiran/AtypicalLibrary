//
// Created by 赵丹 on 25-6-12.
//

#ifndef TENSOR_H
#define TENSOR_H

#include <cstdint>

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
    int32_t device_id;
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
    int64_t* shape;
    int64_t* strides;
    DLDataType dtype;
    Device device;
};

#endif//TENSOR_H
