//
// Created by 赵丹 on 25-6-24.
//

#ifndef TENSOR_UTILS_H
#define TENSOR_UTILS_H

#include <cstdint>
#include <stdexcept>
#include <string>

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

#if defined(__clang__)
#define __ubsan_ignore_pointer_overflow__ __attribute__((no_sanitize("pointer-overflow")))
#else
#define __ubsan_ignore_pointer_overflow__
#endif

#if defined(__GNUC__)
#define ATTRIBUTE_UNUSED __attribute__((unused))
#else
#define ATTRIBUTE_UNUSED
#endif


#define UNUSED(expr)   \
    do {               \
        (void) (expr); \
    } while (false)

#define STR_CONCAT_(__x, __y) __x##__y
#define STR_CONCAT(__x, __y) STR_CONCAT_(__x, __y)

#define REG_VAR_DEF static ATTRIBUTE_UNUSED uint32_t _make_unique_tid_

namespace atp {

enum class DeviceType : uint8_t {
    kCPU = 0,
    kCUDA = 1,
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

}// namespace atp

#endif//TENSOR_UTILS_H
