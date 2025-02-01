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

private:
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

public:
    // Information for object protocol.
    static constexpr uint32_t _type_index = static_cast<uint32_t>(TypeIndex::kRuntimeNDArray);
    static constexpr uint32_t _type_child_slots = 0;
    static constexpr uint32_t _type_child_slots_can_overflow = true;
    static constexpr const char* _type_key = "runtime.NDArray";

    TVM_DECLARE_BASE_OBJECT_INFO(NDArrayNode, Object);
};

class NDArray : public ObjectRef {
public:
    /*! \brief Container type for Object system. */
    using ContainerType = NDArrayNode;

    /*! \brief default constructor */
    NDArray() = default;



    /*! \return Whether the tensor is contiguous */
    inline bool IsContiguous() const;

protected:
    /*!
   * \brief Get mutable internal container pointer.
   * \return a mutable container pointer.
   */
    NODISCARD ContainerType* get_mutable() const {
        return static_cast<ContainerType*>(data_.get());
    }

};

// implementations of inline functions
/*!
 * \brief return the size of data the DLTensor hold, in terms of number of bytes
 *
 *  \param arr the input DLTensor
 *  \return number of bytes of data in the DLTensor.
 */
inline size_t GetDataSize(const DLTensor& arr) {
    size_t size = 1;
    for (tvm_index_t i = 0; i < arr.ndim; ++i) {
        size *= static_cast<size_t>(arr.shape[i]);
    }
    size *= (arr.dtype.bits * arr.dtype.lanes + 7) / 8;
    return size;
}

/*!
 * \brief check if a DLTensor is contiguous.
 * \param arr The input DLTensor.
 * \return The check result.
 */
static bool IsContiguous(const DLTensor& arr) {
    if (arr.strides == nullptr) return true;
    int64_t expected_stride = 1;
    for (int32_t i = arr.ndim; i != 0; --i) {
        int32_t k = i - 1;
        if (arr.shape[k] == 1) {
            // Skip stride check if shape[k] is 1, where the dimension is contiguous
            // regardless of the value of stride.
            //
            // For example, PyTorch will normalize stride to 1 if shape is 1 when exporting
            // to DLPack.
            // More context: https://github.com/pytorch/pytorch/pull/83158
            continue;
        }
        if (arr.strides[k] != expected_stride) return false;
        expected_stride *= arr.shape[k];
    }
    return true;
}



}// namespace litetvm::runtime

#endif//NDARRAY_H
