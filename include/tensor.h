//
// Created by 赵丹 on 25-6-12.
//

#ifndef TENSOR_H
#define TENSOR_H

#include "tensor_impl.h"

namespace atp {

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

// TODO: bfloat, float8, float6 and float4 type need to be defined.
#define SCALAR_TYPES_TO_CPP_TYPES(f)           \
    f(DLDataTypeCode::kInt, 8, 1, int8_t);     \
    f(DLDataTypeCode::kInt, 16, 1, int16_t);   \
    f(DLDataTypeCode::kInt, 32, 1, int32_t);   \
    f(DLDataTypeCode::kInt, 64, 1, int64_t);   \
    f(DLDataTypeCode::kUInt, 8, 1, uint8_t);   \
    f(DLDataTypeCode::kUInt, 16, 1, uint16_t); \
    f(DLDataTypeCode::kUInt, 32, 1, uint32_t); \
    f(DLDataTypeCode::kUInt, 64, 1, uint64_t); \
    f(DLDataTypeCode::kBool, 8, 1, bool);      \
    f(DLDataTypeCode::kFloat, 32, 1, float);   \
    f(DLDataTypeCode::kFloat, 64, 1, double);  \
    // f(DLDataTypeCode::kBfloat, 16, 1, float);            \
    // f(DLDataTypeCode::kFloat8_e3m4, 8, 1, float);        \
    // f(DLDataTypeCode::kFloat8_e4m3, 8, 1, float);        \
    // f(DLDataTypeCode::kFloat8_e4m3b11fnuz, 8, 1, float); \
    // f(DLDataTypeCode::kFloat8_e4m3fn, 8, 1, float);      \
    // f(DLDataTypeCode::kFloat8_e4m3fnuz, 8, 1, float);    \
    // f(DLDataTypeCode::kFloat8_e5m2, 8, 1, float);        \
    // f(DLDataTypeCode::kFloat8_e5m2fnuz, 8, 1, float);    \
    // f(DLDataTypeCode::kFloat8_e8m0fnu, 8, 1, float);     \
    // f(DLDataTypeCode::kFloat6_e2m3fn, 6, 1, float);      \
    // f(DLDataTypeCode::kFloat6_e3m2fn, 6, 1, float);      \
    // f(DLDataTypeCode::kFloat4_e2m1fn, 4, 1, float)


inline std::string Type2Str(const DLDataType& dtype) {
    switch (dtype.code) {
        case DLDataTypeCode::kInt: {
            return "int";
        }

        case DLDataTypeCode::kUInt: {
            return "uint";
        }

        case DLDataTypeCode::kFloat: {
            return "float";
        }

        case DLDataTypeCode::kBfloat: {
            return "bfloat";
        }

        case DLDataTypeCode::kFloat8_e3m4: {
            return "float8_e3m4";
        }

        case DLDataTypeCode::kFloat8_e4m3: {
            return "float8_e4m3";
        }

        case DLDataTypeCode::kFloat8_e4m3b11fnuz: {
            return "float8_e4m3b11fnuz";
        }

        case DLDataTypeCode::kFloat8_e4m3fn: {
            return "float8_e4m3fn";
        }

        case DLDataTypeCode::kFloat8_e4m3fnuz: {
            return "float8_e4m3fnuz";
        }

        case DLDataTypeCode::kFloat8_e5m2: {
            return "float8_e5m2";
        }

        case DLDataTypeCode::kFloat8_e5m2fnuz: {
            return "float8_e5m2fnuz";
        }

        case DLDataTypeCode::kFloat8_e8m0fnu: {
            return "float8_e8m0fnu";
        }

        case DLDataTypeCode::kFloat6_e2m3fn: {
            return "float6_e2m3fn";
        }

        case DLDataTypeCode::kFloat6_e3m2fn: {
            return "float6_e3m2fn";
        }

        case DLDataTypeCode::kFloat4_e2m1fn: {
            return "float4_e2m1fn";
        }

        default: {
            throw std::runtime_error("Unsupported data type");
        }
    }
}

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

class Tensor {
public:
    Tensor() = default;

    explicit Tensor(const std::vector<int64_t>& shape,
                    DeviceType device_type = DeviceType::kCPU,
                    DLDataType dtype = {DLDataTypeCode::kFloat, 32, 1});


    Tensor(const Tensor&) = default;
    Tensor(Tensor&&) = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor& operator=(Tensor&&) = default;

    // returns a tensor filled with random numbers from
    // a uniform distribution on the interval [0, 1)
    static Tensor rand(const std::vector<int64_t>& shape);

    // Returns a tensor filled with random numbers from
    // a normal distribution with mean 0 and variance 1
    static Tensor randn(const std::vector<int64_t>& shape);

    static Tensor randint(int64_t low, int64_t high, const std::vector<int64_t>& shape);

    NODISCARD bool defined() const;

    NODISCARD int32_t use_count() const;

    NODISCARD bool unique() const;

    // NODISCARD void* data() const;

    NODISCARD void* data_ptr() const;

    NODISCARD const void* const_data_ptr() const;

    NODISCARD std::vector<int64_t> shape() const;

    NODISCARD DLDataType dtype() const;

    NODISCARD int32_t ndim() const;

    NODISCARD int64_t numel() const;

    NODISCARD int64_t nbytes() const;

    NODISCARD Scalar item() const;

    template<typename T>
    T* data_ptr() const;

    template<typename T,
             std::enable_if_t<!std::is_const_v<T>>* = nullptr>
    const T* const_data_ptr() const;

    template<typename T,
             std::enable_if_t<std::is_const_v<T>>* = nullptr>
    const std::remove_const_t<T>* const_data_ptr() const;

private:
    std::shared_ptr<TensorImpl> data_;
    int64_t byte_offset_{0};
};

std::ostream& operator<<(std::ostream& os, const Tensor& t);

}// namespace atp


#endif//TENSOR_H
