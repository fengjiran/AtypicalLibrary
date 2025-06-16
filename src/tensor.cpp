//
// Created by 赵丹 on 25-6-12.
//

#include "tensor.h"

namespace atp {

class Tensor::TensorNode {
public:
    TensorNode() = default;

    TensorNode(std::vector<int64_t> shape, DeviceType device_type, DLDataType dtype) {
        tensor_info_.ndim = static_cast<int32_t>(shape.size());
        tensor_info_.shape = std::move(shape);
        tensor_info_.device_type = device_type;
        tensor_info_.dtype = dtype;
        tensor_info_.strides.resize(tensor_info_.ndim, 1);
        for (int i = tensor_info_.ndim - 2; i >= 0; --i) {
            tensor_info_.strides[i] = tensor_info_.strides[i + 1] * tensor_info_.shape[i + 1];
        }

        if (device_type == DeviceType::kCPU) {
            alloc_ = new CPUAllocator;
        } else if (device_type == DeviceType::kCUDA) {
            alloc_ = new CUDAAllocator;
        } else {
            throw std::runtime_error("Unsupported device type");
        }

        tensor_info_.data = alloc_->allocate(GetTensorSize(tensor_info_));
    }

    ~TensorNode() {
        alloc_->deallocate(tensor_info_.data);
        delete alloc_;
    }

    NODISCARD void* data() const {
        return tensor_info_.data;
    }

    NODISCARD std::vector<int64_t> shape() const {
        return tensor_info_.shape;
    }

    NODISCARD DLDataType dtype() const {
        return tensor_info_.dtype;
    }

private:
    TensorInfo tensor_info_{};
    Allocator* alloc_{nullptr};
};

Tensor::Tensor(const std::vector<int64_t>& shape, DeviceType device_type, DLDataType dtype) {
    data_ = std::make_shared<TensorNode>(shape, device_type, dtype);
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


}// namespace atp