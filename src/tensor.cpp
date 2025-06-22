//
// Created by 赵丹 on 25-6-12.
//

#include "tensor.h"
#include <glog/logging.h>
#include <random>

namespace atp {

Tensor::Tensor(const std::vector<int64_t>& shape, DeviceType device_type, DLDataType dtype) {
    data_ = std::make_shared<TensorImpl>(shape, device_type, dtype);
}

Tensor Tensor::rand(const std::vector<int64_t>& shape) {
    Tensor t(shape);
    CHECK(t.numel() > 0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0, 1);
    for (int64_t i = 0; i < t.numel(); ++i) {
        static_cast<float*>(t.data())[i] = dist(gen);
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
        static_cast<float*>(t.data())[i] = dist(gen);
    }
    return t;
}

Tensor Tensor::randint(int64_t low, int64_t high, const std::vector<int64_t>& shape) {
    Tensor t(shape, DeviceType::kCPU, {DLDataTypeCode::kInt, 64, 1});
    CHECK(t.numel() > 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int64_t> dist(low, high);
    for (int64_t i = 0; i < t.numel(); ++i) {
        static_cast<int64_t*>(t.data())[i] = dist(gen);
    }
    return t;
}


bool Tensor::defined() const {
    return data_ != nullptr;
}


int32_t Tensor::use_count() const {
    return static_cast<int32_t>(data_.use_count());
}

bool Tensor::unique() const {
    return data_.use_count() == 1;
}

void* Tensor::data() const {
    if (data_) {
        return data_->data();
    }
    return nullptr;
}


std::vector<int64_t> Tensor::shape() const {
    return data_->shape();
}

DLDataType Tensor::dtype() const {
    return data_->dtype();
}

int32_t Tensor::ndim() const {
    return data_->ndim();
}

int64_t Tensor::numel() const {
    return data_->numel();
}

int64_t Tensor::nbytes() const {
    return data_->nbytes();
}

namespace {
void check_type(const Tensor& t, DLDataTypeCode type_code, int8_t type_bits, int16_t type_lanes) {
    CHECK(t.dtype().code == type_code && t.dtype().bits == type_bits && t.dtype().lanes == type_lanes)
            << "data type mismatch.";
}
}// namespace


#define DEFINE_DATA_PTR(type_code, type_bits, type_lanes, T)         \
    template<>                                                       \
    const T* Tensor::const_data_ptr<T>() const {                     \
        check_type(*this, type_code, type_bits, type_lanes);         \
        return data_->const_data_ptr_impl<T>();                      \
    }                                                                \
                                                                     \
    template<>                                                       \
    const T* Tensor::const_data_ptr<const T>() const {               \
        check_type(*this, type_code, type_bits, type_lanes);         \
        return data_->const_data_ptr_impl<T>(); \
    }

SCALAR_TYPES_TO_CPP_TYPES(DEFINE_DATA_PTR);

#undef DEFINE_DATA_PTR


}// namespace atp