//
// Created by 赵丹 on 25-6-24.
//

#ifndef TENSOR_UTILS_H
#define TENSOR_UTILS_H

#include "macros.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace atp {

enum class DeviceType : uint8_t {
    kCPU = 0,
    kCUDA = 1,
    kUndefined
};

struct Device {
    DeviceType device_type;
    uint8_t device_id;
};

inline std::string DeviceType2Str(const DeviceType& device_type) {
    switch (device_type) {
        case DeviceType::kCPU: {
            return "CPU";
        }

        case DeviceType::kCUDA: {
            return "CUDA";
        }

        default: {
            throw std::runtime_error("Unsupported device type");
        }
    }
}

inline bool mul_overflow(int64_t a, int64_t b, int64_t* out) {
    return __builtin_mul_overflow(a, b, out);
}

inline bool mul_overflow(uint64_t a, uint64_t b, uint64_t* out) {
    return __builtin_mul_overflow(a, b, out);
}

template<typename Iter>
bool safe_multiply_u64(Iter first, Iter last, uint64_t* out) {
    uint64_t prod = 1;
    bool overflowed = false;
    while (first != last) {
        overflowed |= mul_overflow(prod, *first, &prod);
        ++first;
    }
    *out = prod;
    return overflowed;
}

template<typename Container>
bool safe_multiply_u64(const Container& c, uint64_t* out) {
    return safe_multiply_u64(c.begin(), c.end(), out);
}

}// namespace atp

#endif//TENSOR_UTILS_H
