//
// Created by 赵丹 on 25-6-12.
//

#include "tensor.h"
#include <glog/logging.h>
#include <random>

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

    const void* const_data_ptr() const {
        return tensor_info_.data;
    }

    NODISCARD std::vector<int64_t> shape() const {
        return tensor_info_.shape;
    }

    NODISCARD DLDataType dtype() const {
        return tensor_info_.dtype;
    }

    NODISCARD int32_t ndim() const {
        return tensor_info_.ndim;
    }

    NODISCARD int64_t element_size() const {
        return (tensor_info_.dtype.bits * tensor_info_.dtype.lanes + 7) / 8;
    }

    NODISCARD int64_t numel() const {
        if (tensor_info_.shape.empty()) {
            return 0;
        }

        int64_t numel = 1;
        for (int64_t dim: tensor_info_.shape) {
            numel *= dim;
        }
        return numel;
    }

    NODISCARD int64_t nbytes() const {
        return GetTensorSize(tensor_info_);
    }


private:
    TensorInfo tensor_info_{};
    Allocator* alloc_{nullptr};
};

Tensor::Tensor(const std::vector<int64_t>& shape, DeviceType device_type, DLDataType dtype) {
    data_ = std::make_shared<TensorNode>(shape, device_type, dtype);
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

const void* Tensor::const_data_ptr() const {
    if (data_) {
        return data_->const_data_ptr();
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


}// namespace atp