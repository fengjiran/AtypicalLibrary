//
// Created by richard on 1/31/25.
//

#ifndef NDARRAY_H
#define NDARRAY_H

#include "runtime/c_runtime_api.h"
#include "runtime/object.h"
#include "shape_tuple.h"

// nested namespace
namespace litetvm::runtime {

using Device = DLDevice;

class NDArrayNode : public Object {
public:
    NDArrayNode() {
        type_index_ = RuntimeTypeIndex();
        dl_tensor.data = nullptr;
        dl_tensor.ndim = 0;
        dl_tensor.shape = nullptr;
        dl_tensor.strides = nullptr;
        dl_tensor.byte_offset = 0;
    }

  NDArrayNode(void* data, ShapeTuple shape, DLDataType dtype, Device dev) {
      type_index_ = RuntimeTypeIndex();
      shape_ = std::move(shape);
      dl_tensor.data = data;
      dl_tensor.ndim = static_cast<int>(shape_.size());
      dl_tensor.shape = const_cast<ShapeTuple::index_type*>(shape_.data());
      dl_tensor.dtype = dtype;
      dl_tensor.strides = nullptr;
      dl_tensor.byte_offset = 0;
      dl_tensor.device = dev;
    }


    /*!
   * \brief The corresponding dl_tensor field.
   * \note it is important that the first field is DLTensor
   *  So that this data structure is DLTensor compatible.
   *  The head ptr of this struct can be viewed as DLTensor*.
   */
    DLTensor dl_tensor;
    /*!
     * \brief additional context, reserved for recycling
     * \note We can attach additional content here
     *  which the current container depend on
     *  (e.g. reference to original memory when creating views).
     */
    void* manager_ctx{nullptr};

    /*!
   * \brief The shape container,
   *  can be used for shape data.
   */
    ShapeTuple shape_;

    // Information for object protocol.
    static constexpr uint32_t _type_index = static_cast<uint32_t>(TypeIndex::kRuntimeNDArray);
    static constexpr uint32_t _type_child_slots = 0;
    static constexpr uint32_t _type_child_slots_can_overflow = true;
    static constexpr const char* _type_key = "runtime.NDArray";

    TVM_DECLARE_BASE_OBJECT_INFO(NDArrayNode, Object);
};

}// namespace litetvm::runtime

#endif//NDARRAY_H
