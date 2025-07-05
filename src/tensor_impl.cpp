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


}// namespace atp
