//
// Created by richard on 6/22/25.
//

#include "tensor_impl.h"

namespace atp {

TensorImpl::TensorImpl(const std::vector<int64_t>& shape, int64_t storage_offset, DataType dtype, DeviceType device)
    : storage_offset_(storage_offset), dtype_(dtype) {
    init_bitfield();
    for (int64_t x: shape) {
        numel_ *= x;
    }

    int64_t nbytes = numel_ * dtype_.nbytes();
    storage_ = Storage(nbytes, AllocatorTable::Global().get_allocator(device));
    auto ndim = shape.size();
    shape_and_stride_.set_shape(shape);

    for (int i = ndim - 1; i >= 0; --i) {
        if (i == ndim - 1) {
            shape_and_stride_.stride_at_uncheck(i) = 1;
        } else {
            auto stride = shape_and_stride_.stride_at_uncheck(i + 1) * shape_and_stride_.shape_at_uncheck(i + 1);
            shape_and_stride_.stride_at_uncheck(i) = stride;
        }
    }
}

TensorImpl::TensorImpl(Storage&& storage, DataType dtype, std::optional<DeviceType> device_opt)
    : storage_(std::move(storage)), numel_(0), dtype_(dtype), device_opt_(device_opt) {
    init_bitfield();
}

TensorImpl::TensorImpl(DataType dtype, std::optional<DeviceType> device_opt)
    : TensorImpl({}, dtype, device_opt) {}

TensorImpl::TensorImpl(Storage&& storage, DataType dtype)
    : TensorImpl(std::move(storage), dtype, storage.device()) {}


int64_t TensorImpl::numel() const {
    return numel_;
}

bool TensorImpl::empty() const {
    return numel_ == 0;
}

int64_t TensorImpl::ndim() const {
    return shape_and_stride_.size();
}

std::vector<int64_t> TensorImpl::shape() const {
    return shape_and_stride_.get_shape();
}

std::vector<int64_t> TensorImpl::strides() const {
    return shape_and_stride_.get_strides();
}

size_t TensorImpl::itemsize() const {
    CHECK(dtype_initialized()) << "Can't get item sizer of Tensor that doesn't have initialized dtype.";
    return dtype_.nbytes();
}

bool TensorImpl::has_storage() const {
    return storage_;
}

const Storage& TensorImpl::storage() const {
    return storage_;
}

bool TensorImpl::storage_initialized() const {
    CHECK(has_storage()) << "Can't call storage_initialized() on a tensor that doesn't have storage.";
    return storage_.const_data() != nullptr || numel_ == 0;
}

bool TensorImpl::dtype_initialized() const {
    return dtype_ != DataType();
}

int64_t TensorImpl::storage_offset() const {
    return storage_offset_;
}

DeviceType TensorImpl::device() const {
    CHECK(device_opt_.has_value()) << "tensor does not have a device.";
    return *device_opt_;
}

bool TensorImpl::is_cpu() const {
    return device_opt_.has_value() && device_opt_.value() == DeviceType::kCPU;
}


DataType TensorImpl::dtype() const {
    return dtype_;
}

bool TensorImpl::is_contiguous() const {
    return is_contiguous_;
}


void TensorImpl::set_shape_and_strides(const std::vector<int64_t>& shape, const std::vector<int64_t>& strides, std::optional<int64_t> storage_offset) {
    CHECK(shape.size() == strides.size()) << "dimensionality of shape must match dimensionality of strides.";
    auto ndim = shape.size();
    shape_and_stride_.set_shape(shape);

    bool overflowed = false;
    if (ndim > 0) {
        for (size_t i = ndim - 1;; --i) {
            if (strides[i] >= 0) {
                shape_and_stride_.stride_at_uncheck(i) = strides[i];
            } else {
                if (i == ndim - 1) {
                    shape_and_stride_.stride_at_uncheck(i) = 1;
                } else {
                    overflowed |= mul_overflow(
                            shape_and_stride_.stride_at_uncheck(i + 1),
                            std::max<int64_t>(shape_and_stride_.shape_at_uncheck(i + 1), 1),
                            &shape_and_stride_.stride_at_uncheck(i));
                }
            }

            if (i == 0) {
                break;
            }
        }
        CHECK(!overflowed) << "stride calculation overflowed.";
    }

    refresh_numel();
    refresh_contiguous();
    if (storage_offset.has_value()) {
        storage_offset_ = storage_offset.value();
    }
}

int64_t TensorImpl::safe_compute_numel() const {
    uint64_t numel = 1;
    bool overflow = safe_multiply_u64(shape_and_stride_.get_shape(), &numel);
    constexpr auto numel_max = std::min(
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
            std::numeric_limits<size_t>::max());
    overflow |= numel > numel_max;
    CHECK(!overflow) << "interger multiplication overflow when compute numel.";
    return static_cast<int64_t>(numel);
}

void TensorImpl::refresh_numel() {
    numel_ = safe_compute_numel();
}

bool TensorImpl::compute_contiguous() const {
    return _compute_contiguous(shape(), strides());
}

void TensorImpl::set_contiguous(bool b) {
    is_contiguous_ = b;
}

void TensorImpl::refresh_contiguous() {
    set_contiguous(compute_contiguous());
}

void TensorImpl::set_storage_offset(int64_t storage_offset) {
    CHECK(storage_offset >= 0) << "storage_offset must be non-negative.";
    storage_offset_ = storage_offset;
}

void* TensorImpl::data() const {
    return data_impl<void>(
            [this] {
                return static_cast<char*>(storage_.data());
            });
}

const void* TensorImpl::const_data() const {
    return data_impl<const void>(
            [this] {
                return static_cast<const char*>(storage_.const_data());
            });
}

void TensorImpl::init_bitfield() {
    is_contiguous_ = true;
}


}// namespace atp
