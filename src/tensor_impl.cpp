//
// Created by richard on 6/22/25.
//

#include "tensor_impl.h"

namespace atp {

TensorImpl::TensorImpl(std::vector<int64_t> shape, DeviceType device_type, DLDataType dtype)
    : alloc_(AllocatorTable::Global().get_allocator(device_type)) {
    tensor_info_.ndim = static_cast<int32_t>(shape.size());
    tensor_info_.shape = std::move(shape);
    tensor_info_.device_type = device_type;
    tensor_info_.dtype = dtype;
    tensor_info_.strides.resize(tensor_info_.ndim, 1);
    for (int i = tensor_info_.ndim - 2; i >= 0; --i) {
        tensor_info_.strides[i] = tensor_info_.strides[i + 1] * tensor_info_.shape[i + 1];
    }

    data_ptr_ = alloc_->allocate_bk(GetTensorSize(tensor_info_));
    tensor_info_.data = data_ptr_.get();
}

TensorImpl::~TensorImpl() = default;

void* TensorImpl::data_ptr() const {
    auto get_data = [this] {
        return static_cast<char*>(tensor_info_.data);
    };
    return data_impl<void>(get_data);
}

const void* TensorImpl::const_data_ptr() const {
    auto get_data = [this] {
        return static_cast<const char*>(tensor_info_.data);
    };
    return data_impl<const void>(get_data);
}


std::vector<int64_t> TensorImpl::shape() const {
    return tensor_info_.shape;
}

DLDataType TensorImpl::dtype() const {
    return tensor_info_.dtype;
}

int32_t TensorImpl::ndim() const {
    return tensor_info_.ndim;
}

int64_t TensorImpl::element_size() const {
    return (tensor_info_.dtype.bits * tensor_info_.dtype.lanes + 7) / 8;
}

int64_t TensorImpl::numel() const {
    if (tensor_info_.shape.empty()) {
        return 0;
    }

    int64_t numel = 1;
    for (int64_t dim: tensor_info_.shape) {
        numel *= dim;
    }
    return numel;
}

int64_t TensorImpl::nbytes() const {
    return GetTensorSize(tensor_info_);
}

// Scalar TensorImpl::item() const {
//     CHECK(numel() == 1) << fmt::format("a Tensor with {} elements can not converted to Scalar.", numel());
// }


}// namespace atp
