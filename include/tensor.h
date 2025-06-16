//
// Created by 赵丹 on 25-6-12.
//

#ifndef TENSOR_H
#define TENSOR_H

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

class Allocator {
public:
    explicit Allocator(DeviceType device) : device_(device) {}

    virtual ~Allocator() = default;

    NODISCARD DeviceType device() const {
        return device_;
    }

    NODISCARD virtual void* allocate(size_t n) const = 0;

    virtual void deallocate(void* p) const = 0;

private:
    DeviceType device_;
};

class CPUAllocator : public Allocator {
public:
    CPUAllocator() : Allocator(DeviceType::kCPU) {}

    NODISCARD void* allocate(size_t n) const override {
        return malloc(n);
    }

    void deallocate(void* p) const override {
        free(p);
    }
};

class CUDAAllocator : public Allocator {
public:
    CUDAAllocator() : Allocator(DeviceType::kCUDA) {}
    NODISCARD void* allocate(size_t n) const override {
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

inline size_t GetTensorSize(const TensorInfo& t) {
    if (t.shape.empty()) {
        return 0;
    }

    size_t numel = 1;
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

    NODISCARD int32_t use_count() const;

    NODISCARD bool unique() const;

    NODISCARD void* data() const;

    NODISCARD std::vector<int64_t> shape() const;

    NODISCARD DLDataType dtype() const;


private:
    class TensorNode;
    std::shared_ptr<TensorNode> data_;
};

}// namespace atp


#endif//TENSOR_H
