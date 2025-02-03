//
// Created by richard on 1/31/25.
//

#include "runtime/ndarray.h"
#include "runtime/device_api.h"

static void TVMNDArrayDLPackDeleter(DLManagedTensor* tensor);

namespace litetvm::runtime {

inline void VerifyDataType(DLDataType dtype) {
    CHECK_GE(dtype.lanes, 1);
    if (dtype.code == static_cast<uint8_t>(DLDataTypeCode::kDLFloat)) {
        CHECK_EQ(dtype.bits % 8, 0);
    } else {
        // allow uint1 as a special flag for bool.
        if (dtype.bits == 1 && dtype.code == static_cast<uint8_t>(DLDataTypeCode::kDLUInt))
            return;
        // allow int1/uint4/int4
        if (dtype.bits == 1 && dtype.code == static_cast<uint8_t>(DLDataTypeCode::kDLInt))
            return;
        if (dtype.bits == 4 && dtype.code == static_cast<uint8_t>(DLDataTypeCode::kDLUInt))
            return;
        if (dtype.bits == 4 && dtype.code == static_cast<uint8_t>(DLDataTypeCode::kDLInt))
            return;
        CHECK_EQ(dtype.bits % 8, 0);
    }
    CHECK_EQ(dtype.bits & (dtype.bits - 1), 0);
}

struct NDArray::Internal {
    // Default deleter for the container
    static void DefaultDeleter(Object* ptr_obj) {
        auto* ptr = static_cast<Container*>(ptr_obj);
        if (ptr->manager_ctx != nullptr) {
            static_cast<Container*>(ptr->manager_ctx)->DecRef();
        } else if (ptr->dl_tensor.data != nullptr) {
            DeviceAPI::Get(ptr->dl_tensor.device)
                ->FreeDataSpace(ptr->dl_tensor.device, ptr->dl_tensor.data);
        }
        delete ptr;
    }

    // Deleter for NDArray converted from DLPack
    // This is used from data which is passed from external DLPack(DLManagedTensor)
    // that are not allocated inside of TVM.
    // This enables us to create NDArray from memory allocated by other
    // frameworks that are DLPack compatible
    static void DLPackDeleter(Object* ptr_obj) {
        auto* ptr = static_cast<Container*>(ptr_obj);
        auto* tensor = static_cast<DLManagedTensor*>(ptr->manager_ctx);
        if (tensor->deleter != nullptr) {
            (*tensor->deleter)(tensor);
        }
        delete ptr;
    }

    // Deleter for NDArray based on external DLTensor
    // The memory is allocated from outside and it is assumed that
    // responsibility for its freeing is also outside
    static void SelfDeleter(Object* ptr_obj) {
        auto* ptr = static_cast<Container*>(ptr_obj);
        delete ptr;
    }

    // Local create function which allocates tensor metadata
    // but does not allocate space for the data.
    static NDArray Create(ShapeTuple shape, DLDataType dtype, Device dev) {
        VerifyDataType(dtype);

        // critical zone: construct header
        auto* data = new Container();
        data->SetDeleter(DefaultDeleter);

        // RAII now in effect
        NDArray ret(GetObjectPtr<Object>(data));
        // setup shape
        data->shape_ = std::move(shape);
        data->dl_tensor.shape = const_cast<ShapeTuple::index_type*>(data->shape_.data());
        data->dl_tensor.ndim = static_cast<int>(data->shape_.size());
        // setup dtype
        data->dl_tensor.dtype = dtype;
        // setup device
        data->dl_tensor.device = dev;
        return ret;
    }

    // Implementation of API function
    static DLTensor* MoveToFFIHandle(NDArray arr) {
        DLTensor* handle = FFIGetHandle(arr);
        FFIClearAfterMove(&arr);
        return handle;
    }

    static void FFIDecRef(TVMArrayHandle tensor) {
        NDArray::FFIDecRef(tensor);
    }

    // Container to DLManagedTensor
    static DLManagedTensor* ToDLPack(TVMArrayHandle handle) {
        auto* from =
            static_cast<Container*>(reinterpret_cast<ContainerBase*>(handle));
        return ToDLPack(from);
    }

    static DLManagedTensor* ToDLPack(Container* from) {
        CHECK(from != nullptr);
        auto* ret = new DLManagedTensor();
        ret->dl_tensor = from->dl_tensor;
        ret->manager_ctx = from;
        from->IncRef();
        ret->deleter = TVMNDArrayDLPackDeleter;
        return ret;
    }

    // Delete dlpack object.
    static void NDArrayDLPackDeleter(DLManagedTensor* tensor) {
        static_cast<Container*>(tensor->manager_ctx)->DecRef();
        delete tensor;
    }
};

ShapeTuple NDArray::Shape() const {
    return static_cast<const Container*>(data_.get())->shape_;
}

DataType NDArray::DataType() const {
    return runtime::DataType(get_mutable()->dl_tensor.dtype);
}

DLManagedTensor* NDArray::ToDLPack() const {
    return Internal::ToDLPack(get_mutable());
}

}

using namespace litetvm::runtime;
void TVMNDArrayDLPackDeleter(DLManagedTensor* tensor) {
    NDArray::Internal::NDArrayDLPackDeleter(tensor);
}