//
// Created by richard on 2/6/25.
//

#ifndef FUNC_ARGS_H
#define FUNC_ARGS_H

#include "runtime/c_runtime_api.h"
#include "runtime/object.h"

#include <glog/logging.h>

namespace litetvm::runtime {

// forward declarations
class NDArray;
class Module;
class PackedFunc;

/*!
 * \brief Convert argument type code to string.
 * \param type_code The input type code.
 * \return The corresponding string repr.
 */
inline const char* ArgTypeCode2Str(int type_code);

#define TVM_LOG_INCORRECT_TYPE_CODE(CODE, T) \
    "expected " << ArgTypeCode2Str(T) << " but got " << ArgTypeCode2Str(CODE)

// macro to check type code.
#define TVM_CHECK_TYPE_CODE(CODE, T) CHECK_EQ(CODE, T) << TVM_LOG_INCORRECT_TYPE_CODE(CODE, T)


/*!
 * \brief Internal base class to
 *  handle conversion to POD values.
 */
class TVMPODValue_ {
public:
    operator void*() const {
        if (type_code_ == static_cast<int>(TVMArgTypeCode::kTVMNullptr)) {
            return nullptr;
        }

        if (type_code_ == static_cast<int>(TVMArgTypeCode::kTVMDLTensorHandle)) {
            return value_.v_handle;
        }

        TVM_CHECK_TYPE_CODE(type_code_, static_cast<int>(TVMArgTypeCode::kTVMOpaqueHandle));
        return value_.v_handle;
    }

    operator DLTensor*() const {
        if (type_code_ == static_cast<int>(TVMArgTypeCode::kTVMDLTensorHandle) ||
            type_code_ == static_cast<int>(TVMArgTypeCode::kTVMNDArrayHandle)) {
            return static_cast<DLTensor*>(value_.v_handle);
        }

        if (type_code_ == static_cast<int>(TVMArgTypeCode::kTVMNullptr)) {
            return nullptr;
        }

        LOG(FATAL) << "Expected "
                   << "DLTensor* or NDArray but got " << ArgTypeCode2Str(type_code_);
        return nullptr;
    }

    operator NDArray() const {
        if (type_code_ == static_cast<int>(TVMArgTypeCode::kTVMNullptr)) {
            return NDArray(ObjectPtr<Object>(nullptr));
        }

        TVM_CHECK_TYPE_CODE(type_code_, static_cast<int>(TVMArgTypeCode::kTVMNDArrayHandle));
        return NDArray(NDArray::FFIDataFromHandle(static_cast<TVMArrayHandle>(value_.v_handle)));
    }

    operator Module() const {
        if (type_code_ == static_cast<int>(TVMArgTypeCode::kTVMNullptr)) {
            return Module(ObjectPtr<Object>(nullptr));
        }
        TVM_CHECK_TYPE_CODE(type_code_, static_cast<int>(TVMArgTypeCode::kTVMModuleHandle));
        return Module(ObjectPtr<Object>(static_cast<Object*>(value_.v_handle)));
    }

    operator PackedFunc() const {
        if (type_code_ == static_cast<int>(TVMArgTypeCode::kTVMNullptr)) {
            return PackedFunc(ObjectPtr<Object>(nullptr));
        }

        TVM_CHECK_TYPE_CODE(type_code_, static_cast<int>(TVMArgTypeCode::kTVMPackedFuncHandle));
        return PackedFunc(ObjectPtr<Object>(static_cast<Object*>(value_.v_handle)));
    }

    operator Device() const {
        TVM_CHECK_TYPE_CODE(type_code_, static_cast<int>(TVMArgTypeCode::kDLDevice));
        return value_.v_device;
    }

    NODISCARD int type_code() const {
        return type_code_;
    }
    /*!
   * \brief return handle as specific pointer type.
   * \tparam T the data type.
   * \return The pointer type.
   */
    template<typename T>
    T* ptr() const {
        return static_cast<T*>(value_.v_handle);
    }

    NODISCARD std::optional<bool> TryAsBool() const {
        // Helper function to reduce duplication in the variable integer
        // conversions.  This is publicly exposed, as it can be useful in
        // specializations of PackedFuncValueConverter.
        if (type_code_ == static_cast<int>(TVMArgTypeCode::kTVMArgBool)) {
            return static_cast<bool>(value_.v_int64);
        }

        return std::nullopt;
    }

    NODISCARD std::optional<int64_t> TryAsInt() const {
        // Helper function to reduce duplication in the variable integer
        // conversions.  This is publicly exposed, as it can be useful in
        // specializations of PackedFuncValueConverter.
        if (type_code_ == static_cast<int>(DLDataTypeCode::kDLInt)) {
            return value_.v_int64;
        }

        return std::nullopt;
    }

    NODISCARD std::optional<double> TryAsFloat() const {
        // Helper function to reduce duplication in the variable integer
        // conversions.  This is publicly exposed, as it can be useful in
        // specializations of PackedFuncValueConverter.
        if (type_code_ == static_cast<int>(DLDataTypeCode::kDLFloat)) {
            return value_.v_float64;
        }
        return std::nullopt;
    }

protected:
    friend class TVMArgsSetter;
    friend class TVMRetValue;
    friend class TVMMovableArgValue_;

    TVMPODValue_() : type_code_(static_cast<int>(TVMArgTypeCode::kTVMNullptr)) {}

    TVMPODValue_(TVMValue value, int type_code) : value_(value), type_code_(type_code) {}

    /*! \brief The value */
    TVMValue value_{};
    /*! \brief the type code */
    int type_code_;
};


// internal namespace
inline const char* ArgTypeCode2Str(int type_code) {
    switch (type_code) {
        case static_cast<int>(DLDataTypeCode::kDLInt):
            return "int";
        case static_cast<int>(TVMArgTypeCode::kTVMArgBool):
            return "bool";
        case static_cast<int>(DLDataTypeCode::kDLUInt):
            return "uint";
        case static_cast<int>(DLDataTypeCode::kDLFloat):
            return "float";
        case static_cast<int>(TVMArgTypeCode::kTVMStr):
            return "str";
        case static_cast<int>(TVMArgTypeCode::kTVMBytes):
            return "bytes";
        case static_cast<int>(TVMArgTypeCode::kTVMOpaqueHandle):
            return "handle";
        case static_cast<int>(TVMArgTypeCode::kTVMNullptr):
            return "NULL";
        case static_cast<int>(TVMArgTypeCode::kTVMDLTensorHandle):
            return "ArrayHandle";
        case static_cast<int>(TVMArgTypeCode::kTVMDataType):
            return "DLDataType";
        case static_cast<int>(TVMArgTypeCode::kDLDevice):
            return "DLDevice";
        case static_cast<int>(TVMArgTypeCode::kTVMPackedFuncHandle):
            return "FunctionHandle";
        case static_cast<int>(TVMArgTypeCode::kTVMModuleHandle):
            return "ModuleHandle";
        case static_cast<int>(TVMArgTypeCode::kTVMNDArrayHandle):
            return "NDArrayContainer";
        case static_cast<int>(TVMArgTypeCode::kTVMObjectHandle):
            return "Object";
        case static_cast<int>(TVMArgTypeCode::kTVMObjectRValueRefArg):
            return "ObjectRValueRefArg";
        default:
            LOG(FATAL) << "unknown type_code=" << type_code;
    }
    throw;
}
}// namespace litetvm::runtime

#endif//FUNC_ARGS_H
