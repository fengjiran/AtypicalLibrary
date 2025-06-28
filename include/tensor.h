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

class Tensor {
public:
    Tensor() = default;

    explicit Tensor(const std::vector<int64_t>& shape,
                    int64_t byte_offset = 0,
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

    NODISCARD std::vector<int64_t> shape() const;

    NODISCARD DLDataType dtype() const;

    NODISCARD int32_t ndim() const;

    NODISCARD int64_t numel() const;

    NODISCARD int64_t nbytes() const;

    NODISCARD Scalar item() const;

    NODISCARD void* data_ptr() const;

    NODISCARD const void* const_data_ptr() const;

    template<typename T>
    T* data_ptr() const;

    template<typename T,
             std::enable_if_t<!std::is_const_v<T>>* = nullptr>
    const T* const_data_ptr() const;

    template<typename T,
             std::enable_if_t<std::is_const_v<T>>* = nullptr>
    const std::remove_const_t<T>* const_data_ptr() const;

private:
    std::shared_ptr<TensorImpl_bk> data_;
    int64_t byte_offset_{0};
};

std::ostream& operator<<(std::ostream& os, const Tensor& t);

}// namespace atp


#endif//TENSOR_H
