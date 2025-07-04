//
// Created by 赵丹 on 25-6-12.
//

#include "tensor.h"
#include <glog/logging.h>
#include <random>

namespace atp {

namespace {
void check_type(const Tensor& t, DLDataTypeCode type_code, int8_t type_bits, int16_t type_lanes) {
    CHECK(t.dtype() == DataType(type_code, type_bits, type_lanes));
    // CHECK(t.dtype().code() == type_code && t.dtype().bits() == type_bits && t.dtype().lanes() == type_lanes)
    //         << "data type mismatch.";
}
}// namespace

#define DEFINE_DATA_PTR(type_code, type_bits, type_lanes, name, T) \
    template<>                                                     \
    T* Tensor::data_ptr<T>() const {                               \
        check_type(*this, type_code, type_bits, type_lanes);       \
        return impl_->data_ptr_impl<T>();                          \
    }                                                              \
                                                                   \
    template<>                                                     \
    const T* Tensor::const_data_ptr<T>() const {                   \
        check_type(*this, type_code, type_bits, type_lanes);       \
        return impl_->const_data_ptr_impl<T>();                    \
    }                                                              \
                                                                   \
    template<>                                                     \
    const T* Tensor::const_data_ptr<const T>() const {             \
        check_type(*this, type_code, type_bits, type_lanes);       \
        return impl_->const_data_ptr_impl<T>();                    \
    }

SCALAR_TYPE_TO_NAME_AND_CPP_TYPE(DEFINE_DATA_PTR);
#undef DEFINE_DATA_PTR

Tensor::Tensor(const std::vector<int64_t>& shape, int64_t storage_offset, DataType dtype, DeviceType device)
    : impl_(std::make_shared<TensorImpl>(shape, storage_offset, dtype, device)) {}


bool Tensor::defined() const {
    return impl_ != nullptr && impl_->storage_initialized();
}

int32_t Tensor::use_count() const {
    return static_cast<int32_t>(impl_.use_count());
}

bool Tensor::unique() const {
    return impl_.use_count() == 1;
}

void* Tensor::data_ptr() const {
    return impl_ ? impl_->data() : nullptr;
}

const void* Tensor::const_data_ptr() const {
    return impl_ ? impl_->const_data() : nullptr;
}

std::vector<int64_t> Tensor::shape() const {
    return defined() ? impl_->shape() : std::vector<int64_t>{};
}

DataType Tensor::dtype() const {
    return defined() ? impl_->dtype() : DataType();
}

int32_t Tensor::ndim() const {
    return defined() ? impl_->ndim() : 0;
}

int64_t Tensor::numel() const {
    return defined() ? impl_->numel() : 0;
}

// int64_t Tensor::nbytes() const {
//     return data_->nbytes();
// }

Tensor Tensor::rand(const std::vector<int64_t>& shape) {
    Tensor t(shape);
    CHECK(t.numel() > 0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0, 1);
    for (int64_t i = 0; i < t.numel(); ++i) {
        t.data_ptr<float>()[i] = dist(gen);
        // static_cast<float*>(t.data())[i] = dist(gen);
    }

    return t;
}

Tensor Tensor::randn(const std::vector<int64_t>& shape) {
    Tensor t(shape);
    CHECK(t.numel() > 0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0, 1);
    for (int64_t i = 0; i < t.numel(); ++i) {
        t.data_ptr<float>()[i] = dist(gen);
        // static_cast<float*>(t.data())[i] = dist(gen);
    }
    return t;
}

Tensor Tensor::randint(int64_t low, int64_t high, const std::vector<int64_t>& shape) {
    Tensor t(shape, 0, {DLDataTypeCode::kInt, 64, 1}, DeviceType::kCPU);
    CHECK(t.numel() > 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int64_t> dist(low, high);
    for (int64_t i = 0; i < t.numel(); ++i) {
        t.data_ptr<int64_t>()[i] = dist(gen);
        // static_cast<int64_t*>(t.data())[i] = dist(gen);
    }
    return t;
}

}// namespace atp