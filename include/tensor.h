//
// Created by 赵丹 on 25-6-12.
//

#ifndef TENSOR_H
#define TENSOR_H

#include <csignal>
#include <cstdint>
#include <memory>
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

enum class DeviceType : uint8_t {
    kCPU = 0,
    kCUDA = 1,
};

enum class DLDataTypeCode : uint8_t {
    kInt = 0,
    kUInt = 1,
    kFloat,
    kBfloat,
    kFloat8_e3m4,
    kFloat8_e4m3,
    kFloat8_e4m3b11fnuz,
    kFloat8_e4m3fn,
    kFloat8_e4m3fnuz,
    kFloat8_e5m2,
    kFloat8_e5m2fnuz,
    kFloat8_e8m0fnu,
    kFloat6_e2m3fn,
    kFloat6_e3m2fn,
    kFloat4_e2m1fn
};

struct Device {
    DeviceType device_type;
    uint8_t device_id;
};

/*!
 * \brief The data type the tensor can hold. The data type is assumed to follow the
 * native endian-ness. An explicit error message should be raised when attempting to
 * export an array with non-native endianness
 *
 *  Examples
 *   - float: type_code = 2, bits = 32, lanes = 1
 *   - float4(vectorized 4 float): type_code = 2, bits = 32, lanes = 4
 *   - int8: type_code = 0, bits = 8, lanes = 1
 *   - std::complex<float>: type_code = 5, bits = 64, lanes = 1
 *   - bool: type_code = 6, bits = 8, lanes = 1 (as per common array library convention, the underlying storage size of bool is 8 bits)
 *   - float8_e4m3: type_code = 8, bits = 8, lanes = 1 (packed in memory)
 *   - float6_e3m2fn: type_code = 16, bits = 6, lanes = 1 (packed in memory)
 *   - float4_e2m1fn: type_code = 17, bits = 4, lanes = 1 (packed in memory)
 *
 *  When a sub-byte type is packed, DLPack requires the data to be in little bit-endian, i.e.,
 *  for a packed data set D ((D >> (i * bits)) && bit_mask) stores the i-th element.
 */
struct DLDataType {
    // Data type code.
    DLDataTypeCode code;

    // Number of bits per data element.
    // For example, 8 for int8_t, 32 for float.
    int8_t bits;

    // Number of lanes per data element, for vector.
    int16_t lanes;
};

// TODO: bfloat, float8, float6 and float4 type need to be defined.
#define SCALAR_TYPES_TO_CPP_TYPES(f)                     \
    f(DLDataTypeCode::kInt, 8, 1, int8_t);               \
    f(DLDataTypeCode::kInt, 16, 1, int16_t);             \
    f(DLDataTypeCode::kInt, 32, 1, int32_t);             \
    f(DLDataTypeCode::kInt, 64, 1, int64_t);             \
    f(DLDataTypeCode::kUInt, 8, 1, uint8_t);             \
    f(DLDataTypeCode::kUInt, 16, 1, uint16_t);           \
    f(DLDataTypeCode::kUInt, 32, 1, uint32_t);           \
    f(DLDataTypeCode::kUInt, 64, 1, uint64_t);           \
    f(DLDataTypeCode::kFloat, 32, 1, float);             \
    f(DLDataTypeCode::kFloat, 64, 1, double);            \
    f(DLDataTypeCode::kBfloat, 16, 1, float);            \
    f(DLDataTypeCode::kFloat8_e3m4, 8, 1, float);        \
    f(DLDataTypeCode::kFloat8_e4m3, 8, 1, float);        \
    f(DLDataTypeCode::kFloat8_e4m3b11fnuz, 8, 1, float); \
    f(DLDataTypeCode::kFloat8_e4m3fn, 8, 1, float);      \
    f(DLDataTypeCode::kFloat8_e4m3fnuz, 8, 1, float);    \
    f(DLDataTypeCode::kFloat8_e5m2, 8, 1, float);        \
    f(DLDataTypeCode::kFloat8_e5m2fnuz, 8, 1, float);    \
    f(DLDataTypeCode::kFloat8_e8m0fnu, 8, 1, float);     \
    f(DLDataTypeCode::kFloat6_e2m3fn, 6, 1, float);      \
    f(DLDataTypeCode::kFloat6_e3m2fn, 6, 1, float);      \
    f(DLDataTypeCode::kFloat4_e2m1fn, 4, 1, float)


template<DLDataTypeCode, int8_t, int16_t>
struct DataType2CPPType;

#define DataTypeToCPPType_Specialization(type_code, type_bits, type_lanes, cpp_type) \
    template<>                                                                       \
    struct DataType2CPPType<type_code, type_bits, type_lanes> {                      \
        using type = cpp_type;                                                       \
    }

SCALAR_TYPES_TO_CPP_TYPES(DataTypeToCPPType_Specialization);

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

class Allocator {
public:
    explicit Allocator(DeviceType device) : device_(device) {}

    virtual ~Allocator() = default;

    NODISCARD DeviceType device() const {
        return device_;
    }

    NODISCARD virtual void* allocate(int64_t n) const = 0;

    virtual void deallocate(void* p) const = 0;

private:
    DeviceType device_;
};

class CPUAllocator : public Allocator {
public:
    CPUAllocator() : Allocator(DeviceType::kCPU) {}

    NODISCARD void* allocate(int64_t n) const override {
        return malloc(n);
    }

    void deallocate(void* p) const override {
        free(p);
    }
};

class CUDAAllocator : public Allocator {
public:
    CUDAAllocator() : Allocator(DeviceType::kCUDA) {}
    NODISCARD void* allocate(int64_t n) const override {
        void* p = nullptr;
        // CHECK_CUDA(cudaMalloc(&p, n));
        return p;
    }

    void deallocate(void* p) const override {
        // CHECK_CUDA(cudaFree(p));
    }
};

struct TensorInfo {
    void* data{nullptr};
    int32_t ndim{0};
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    DLDataType dtype;
    DeviceType device_type;
};

inline int64_t GetTensorSize(const TensorInfo& t) {
    if (t.shape.empty()) {
        return 0;
    }

    int64_t numel = 1;
    for (int i = 0; i < t.ndim; ++i) {
        numel *= t.shape[i];
    }

    return (numel * t.dtype.bits * t.dtype.lanes + 7) / 8;
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

    NODISCARD bool defined() const;

    NODISCARD int32_t use_count() const;

    NODISCARD bool unique() const;

    NODISCARD void* data() const;

    const void* const_data_ptr() const;

    NODISCARD std::vector<int64_t> shape() const;

    NODISCARD DLDataType dtype() const;

    NODISCARD int32_t ndim() const;

    NODISCARD int64_t numel() const;

    NODISCARD int64_t nbytes() const;


private:
    class TensorNode;
    std::shared_ptr<TensorNode> data_;
};

std::ostream& operator<<(std::ostream& os, const Tensor& t);

}// namespace atp


#endif//TENSOR_H
