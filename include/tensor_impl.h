//
// Created by richard on 6/22/25.
//

#ifndef TENSOR_IMPL_H
#define TENSOR_IMPL_H

#include <csignal>
#include <cstdint>
#include <fmt/format.h>
#include <glog/logging.h>
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

#if defined(__clang__)
#define __ubsan_ignore_pointer_overflow__ __attribute__((no_sanitize("pointer-overflow")))
#else
#define __ubsan_ignore_pointer_overflow__
#endif


#define UNUSED(expr)   \
    do {               \
        (void) (expr); \
    } while (false)

namespace atp {

enum class DeviceType : uint8_t {
    kCPU = 0,
    kCUDA = 1,
};

enum class DLDataTypeCode : uint8_t {
    kInt = 0,
    kUInt = 1,
    kBool,
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

#define SCALAR_TYPES_NAME(f) \
    f(bool, Bool);           \
    f(uint8_t, Byte);        \
    f(int8_t, Char);         \
    f(uint16_t, UShort);     \
    f(int16_t, Short);       \
    f(uint32_t, UInt);       \
    f(int, Int);             \
    f(uint64_t, ULong);      \
    f(int64_t, Long);        \
    f(float, Float);         \
    f(double, Double);

class Scalar {
public:
    Scalar() : Scalar(static_cast<int64_t>(0)) {}

    template<typename T,
             std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>>* = nullptr>
    Scalar(T val) {
        v.i = static_cast<decltype(v.i)>(val);
        dtype = DLDataTypeCode::kInt;
    }

    template<typename T,
             std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
    Scalar(T val) {
        v.d = static_cast<decltype(v.d)>(val);
        dtype = DLDataTypeCode::kFloat;
    }

    template<typename T,
             std::enable_if_t<std::is_same_v<T, bool>>* = nullptr>
    Scalar(T val) {
        v.i = static_cast<decltype(v.i)>(val);
        dtype = DLDataTypeCode::kBool;
    }

    Scalar(const Scalar& other) : v(other.v), dtype(other.dtype) {}

    Scalar(Scalar&& other) noexcept : v(other.v), dtype(other.dtype) {
        other.v.i = 0;
        other.dtype = DLDataTypeCode::kInt;
    }

    Scalar& operator=(const Scalar& other) {
        Scalar tmp(other);
        swap(*this, tmp);
        return *this;
    }

    Scalar& operator=(Scalar&& other) noexcept {
        Scalar tmp(std::move(other));
        swap(*this, tmp);
        return *this;
    }

    NODISCARD bool isIntegral() const {
        return dtype == DLDataTypeCode::kInt;
    }

    NODISCARD bool isFloatingPoint() const {
        return dtype == DLDataTypeCode::kFloat;
    }

    NODISCARD bool isBool() const {
        return dtype == DLDataTypeCode::kBool;
    }

    NODISCARD DLDataTypeCode type() const {
        return dtype;
    }

    friend void swap(Scalar& a, Scalar& b) noexcept {
        std::swap(a.v, b.v);
        std::swap(a.dtype, b.dtype);
    }

#define ACCESSOR(type, name)                                                 \
    type to##name() const {                                                  \
        if (dtype == DLDataTypeCode::kInt || dtype == DLDataTypeCode::kBool) \
            return static_cast<type>(v.i);                                   \
        else if (dtype == DLDataTypeCode::kFloat)                            \
            return static_cast<type>(v.d);                                   \
        else                                                                 \
            throw std::runtime_error("Unsupported data type");               \
    }
    SCALAR_TYPES_NAME(ACCESSOR);
#undef ACCESSOR

    template<typename T>
    T to() const = delete;

private:
    union val {
        int64_t i;
        double d{};
    } v;

    DLDataTypeCode dtype;
};

#define DEFINE_TO(T, name)           \
    template<>                       \
    inline T Scalar::to<T>() const { \
        return to##name();           \
    }
SCALAR_TYPES_NAME(DEFINE_TO);
#undef DEFINE_TO

class TensorImpl {
public:
    TensorImpl() = delete;

    TensorImpl(std::vector<int64_t> shape, DeviceType device_type, DLDataType dtype);

    ~TensorImpl();

    NODISCARD void* data() const;

    NODISCARD std::vector<int64_t> shape() const;

    NODISCARD DLDataType dtype() const;

    NODISCARD int32_t ndim() const;

    NODISCARD int64_t element_size() const;

    NODISCARD int64_t numel() const;

    NODISCARD int64_t nbytes() const;

    NODISCARD void* data_ptr() const;

    NODISCARD const void* const_data_ptr() const;

    // NODISCARD Scalar item() const;

    template<typename T>
    T* data_ptr_impl() const {
        auto get_data = [this] {
            return static_cast<T*>(tensor_info_.data);
        };
        return data_ptr_impl_impl<T>(get_data);
    }

    template<typename T>
    const T* const_data_ptr_impl() const {
        auto get_data = [this] {
            return static_cast<const T*>(tensor_info_.data);
        };
        return data_ptr_impl_impl<const T>(get_data);
    }


private:
    TensorInfo tensor_info_{};
    Allocator* alloc_{nullptr};

    template<typename Void, typename Func>
    Void* data_impl(const Func& get_data) const {
        auto* data = get_data();
        static_assert(sizeof(*data) == 1, "get_data must return a byte-addressed pointer.");
        return data;
    }

    template<typename T, typename Func>
    __ubsan_ignore_pointer_overflow__ T* data_ptr_impl_impl(const Func& get_data) const {
        CHECK(tensor_info_.data != nullptr);
        return get_data();
    }
};

}// namespace atp

#endif//TENSOR_IMPL_H
