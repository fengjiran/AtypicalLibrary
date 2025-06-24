//
// Created by richard on 6/22/25.
//

#ifndef TENSOR_IMPL_H
#define TENSOR_IMPL_H

#include "tensor_utils.h"
#include "storage_impl.h"

// #include <csignal>
#include <fmt/format.h>
#include <glog/logging.h>
#include <memory>
#include <vector>

namespace atp {

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

    // NODISCARD void* data() const;

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
