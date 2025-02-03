//
// Created by richard on 1/30/25.
//

#ifndef PACKED_FUNC_H
#define PACKED_FUNC_H

#include "runtime/c_runtime_api.h"
#include "runtime/data_type.h"
#include "runtime/ndarray.h"
#include "runtime/object.h"

// nested namespace
namespace litetvm::runtime {

// forward declarations
class TVMArgs;
class TVMArgValue;

template<typename FType>
class TypedPackedFunc;

/*! \brief Using static function to output TypedPackedFunc signature */
using FSig = std::string();

/*! \brief Arguments into TVM functions. */
class TVMArgs {
public:
    const TVMValue* values;
    const int* type_codes;
    int num_args;
    /*!
     * \brief constructor
     * \param values The argument values
     * \param type_codes The argument type codes
     * \param num_args number of arguments.
     */
    TVMArgs(const TVMValue* values, const int* type_codes, int num_args)
        : values(values), type_codes(type_codes), num_args(num_args) {}
    /*! \return size of the arguments */
    NODISCARD int size() const {
        return num_args;
    }
    /*!
     * \brief Get i-th argument
     * \param i the index.
     * \return the ith argument.
     */
    inline TVMArgValue operator[](int i) const;
    /*!
     * \brief Get the i-th argument and do proper type checking with detailed error messages.
     * \tparam T The expected type.
     * \param i The index
     * \return The corresponding argument value.
     */
    template<typename T>
    inline T At(int i) const;
};

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
        if (type_code_ == static_cast<uint8_t>(TVMArgTypeCode::kTVMNullptr))
            return nullptr;

        if (type_code_ == static_cast<int>(TVMArgTypeCode::kTVMDLTensorHandle))
            return value_.v_handle;

        TVM_CHECK_TYPE_CODE(type_code_, static_cast<int>(TVMArgTypeCode::kTVMOpaqueHandle));
        return value_.v_handle;
    }

    operator DLTensor*() const {
        if (type_code_ == static_cast<int>(TVMArgTypeCode::kTVMDLTensorHandle) ||
            type_code_ == static_cast<int>(TVMArgTypeCode::kTVMNDArrayHandle)) {
            return static_cast<DLTensor*>(value_.v_handle);
        }

        if (type_code_ == static_cast<int>(TVMArgTypeCode::kTVMNullptr))
            return nullptr;

        LOG(FATAL) << "Expected "
                   << "DLTensor* or NDArray but got " << ArgTypeCode2Str(type_code_);
        return nullptr;
    }

    operator NDArray() const {
        if (type_code_ == static_cast<int>(TVMArgTypeCode::kTVMNullptr))
            return NDArray(ObjectPtr<Object>(nullptr));

        TVM_CHECK_TYPE_CODE(type_code_, (int) TVMArgTypeCode::kTVMNDArrayHandle);
        return NDArray(NDArray::FFIDataFromHandle(static_cast<TVMArrayHandle>(value_.v_handle)));
    }

    // operator Module() const {
    //   if (type_code_ == static_cast<int>(TVMArgTypeCode::kTVMNullptr)) {
    //     return Module(ObjectPtr<Object>(nullptr));
    //   }
    //   TVM_CHECK_TYPE_CODE(type_code_, static_cast<int>(TVMArgTypeCode::kTVMModuleHandle));
    //   return Module(ObjectPtr<Object>(static_cast<Object*>(value_.v_handle)));
    // }

    // operator PackedFunc() const {
    //   if (type_code_ == static_cast<int>(TVMArgTypeCode::kTVMNullptr)) {
    //     return PackedFunc(ObjectPtr<Object>(nullptr));
    //   }
    //
    //   TVM_CHECK_TYPE_CODE(type_code_, static_cast<int>(TVMArgTypeCode::kTVMPackedFuncHandle));
    //   return PackedFunc(ObjectPtr<Object>(static_cast<Object*>(value_.v_handle)));
    // }

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
    //  friend class TVMArgsSetter;
    //  friend class TVMRetValue;
    //  friend class TVMMovableArgValue_;

    TVMPODValue_() : type_code_(static_cast<int>(TVMArgTypeCode::kTVMNullptr)) {}

    TVMPODValue_(TVMValue value, int type_code) : value_(value), type_code_(type_code) {}

    /*! \brief The value */
    TVMValue value_{};
    /*! \brief the type code */
    int type_code_;
};

/*! \brief A utility class that adds methods useful for each POD type
 *
 * These cannot be provided in the base PODValue_ class, because
 * TVMArgValue and TVMRetValue have different semantics for kTVMStr
 * and kTVMBytes.
 *
 * kTVMStr:
 *
 *     For `TVMArgValue`, the active variant is `v_str`, a `const
 *     char*`.  For `TVMRetValue`, the active variant is `v_handle`,
 *     and should be cast from `void*` to `std::string*`.
 *
 * kTVMBytes:
 *
 *     The active variant is `v_handle`, a `void*`.  For
 *     `TVMArgValue`, should be cast to `TVMByteArray*`.  For
 *     `TVMRetValue`, should be cast to `std::string*`.
 *
 * When converting into an `ObjectRef`, a string may be used to build
 * a `tvm::runtime::String`.  Because TVMArgValue and TVMRetValue use
 * different representations for strings, any utility funciton which
 * might attempt a conversion to an `ObjectRef` must be performed
 * within a context that is aware of the derived class.
 */
template<typename Derived>
class TVMPODValue_CRTP_ : public TVMPODValue_ {
public:
    using TVMPODValue_::TVMPODValue_;

    // ObjectRef handling
    template<typename TObjectRef,
             typename = std::enable_if_t<std::is_base_of_v<ObjectRef, TObjectRef>>>
    NODISCARD bool IsObjectRef() const;

    template<typename TObjectRef>
    TObjectRef AsObjectRef() const;

    operator double() const {
        // Allow automatic conversion from int to float
        // This avoids errors when user pass in int from
        // the frontend while the API expects a float.
        if (auto opt = TryAsFloat()) {
            return opt.value();
        }

        if (auto opt = TryAsInt()) {
            return opt.value();
        }

        if (auto opt = TryAsBool()) {
            return opt.value();
        }

        LOG(FATAL) << TVM_LOG_INCORRECT_TYPE_CODE(type_code_, static_cast<int>(TVMArgTypeCode::kDLDevice));
    }

    operator int64_t() const {
        if (auto opt = TryAsInt()) {
            return opt.value();
        }

        if (auto opt = TryAsBool()) {
            return opt.value();
        }

        LOG(FATAL) << TVM_LOG_INCORRECT_TYPE_CODE(type_code_, static_cast<int>(DLDataTypeCode::kDLInt));
    }

    operator uint64_t() const {
        return operator int64_t();
    }

    operator int() const {
        int64_t value = operator int64_t();
        CHECK_LE(value, std::numeric_limits<int>::max());
        CHECK_GE(value, std::numeric_limits<int>::min());
        return value;
    }

    operator bool() const {
        if (auto opt = TryAsBool()) {
            return opt.value();
        }

        if (auto opt = TryAsInt()) {
            return opt.value();
        }

        LOG(FATAL) << TVM_LOG_INCORRECT_TYPE_CODE(type_code_, static_cast<int>(DLDataTypeCode::kDLInt));
    }
};


/*!
 * \brief A single argument value to PackedFunc.
 *  Containing both type_code and TVMValue
 *
 *  Provides utilities to do type cast into other types.
 */
class TVMArgValue : public TVMPODValue_CRTP_<TVMArgValue> {
public:
    /*! \brief default constructor */
    TVMArgValue() = default;
    /*!
     * \brief constructor
     * \param value of the function
     * \param type_code The type code.
     */
    TVMArgValue(TVMValue value, int type_code) : TVMPODValue_CRTP_(value, type_code) {}
    // reuse converter from parent
    using TVMPODValue_CRTP_::operator double;
    using TVMPODValue_CRTP_::operator int64_t;
    using TVMPODValue_CRTP_::operator uint64_t;
    using TVMPODValue_CRTP_::operator int;
    using TVMPODValue_CRTP_::operator bool;
    using TVMPODValue_::operator void*;
    using TVMPODValue_::operator DLTensor*;
    using TVMPODValue_::operator NDArray;
    using TVMPODValue_::operator Device;
    // using TVMPODValue_::operator Module;
    // using TVMPODValue_::operator PackedFunc;
    using TVMPODValue_CRTP_::AsObjectRef;
    using TVMPODValue_CRTP_::IsObjectRef;

    // conversion operator.
    // operator std::string() const {
    //     if (type_code_ == kTVMDataType) {
    //         return DLDataType2String(operator DLDataType());
    //     } else if (type_code_ == kTVMBytes) {
    //         TVMByteArray* arr = static_cast<TVMByteArray*>(value_.v_handle);
    //         return std::string(arr->data, arr->size);
    //     } else if (type_code_ == kTVMStr) {
    //         return std::string(value_.v_str);
    //     } else {
    //         return AsObjectRef<tvm::runtime::String>().operator std::string();
    //     }
    // }
    // template <typename FType>
    // operator TypedPackedFunc<FType>() const {
    //     return TypedPackedFunc<FType>(operator PackedFunc());
    // }

    NODISCARD const TVMValue& value() const {
        return value_;
    }

    template<typename T, typename = typename std::enable_if<std::is_class<T>::value>::type>
    inline operator T() const;
    inline operator DLDataType() const;
    inline operator DataType() const;
};


/*!
 * \brief Internal auxiliary struct for TypedPackedFunc to indicate a movable argument.
 *
 *  We can only construct a movable argument once from a single argument position.
 *  If the argument is passed as RValue reference, the result will be moved.
 *  We should only construct a MovableArg from an argument once,
 *  as the result will can moved.
 *
 * \note For internal development purpose only.
 */
class TVMMovableArgValue_ : public TVMPODValue_CRTP_<TVMMovableArgValue_> {
public:
    TVMMovableArgValue_(TVMValue value, int type_code) : TVMPODValue_CRTP_(value, type_code) {}
    // reuse converter from parent
    using TVMPODValue_CRTP_::operator double;
    using TVMPODValue_CRTP_::operator int64_t;
    using TVMPODValue_CRTP_::operator uint64_t;
    using TVMPODValue_CRTP_::operator int;
    using TVMPODValue_CRTP_::operator bool;
    using TVMPODValue_::operator void*;
    using TVMPODValue_::operator DLTensor*;
    // using TVMPODValue_::operator NDArray;
    using TVMPODValue_::operator Device;
    // using TVMPODValue_::operator Module;
    // using TVMPODValue_::operator PackedFunc;
    // reuse conversion rule from ArgValue.
    operator std::string() const { return AsArgValue().operator std::string(); }

    // template <typename FType>
    // operator TypedPackedFunc<FType>() const {
    //     return TypedPackedFunc<FType>(operator PackedFunc());
    // }

    operator DLDataType() const { return AsArgValue().operator DLDataType(); }
    operator DataType() const { return AsArgValue().operator DataType(); }
    operator TVMArgValue() const { return AsArgValue(); }
    /*!
     * \brief Helper converter function.
     *  Try to move out an argument if possible,
     *  fall back to normal argument conversion rule otherwise.
     */
    template<typename T,
             typename = typename std::enable_if<std::is_base_of<ObjectRef, T>::value>::type>
    inline operator T() const;

private:
    /*! \return The arg value repr of the value. */
    NODISCARD TVMArgValue AsArgValue() const {
        return {value_, type_code_};
    }
};

inline TVMArgValue TVMArgs::operator[](int i) const {
    CHECK_LT(i, num_args) << "not enough argument passed, " << num_args << " passed"
                          << " but request arg[" << i << "].";
    return TVMArgValue(values[i], type_codes[i]);
}

template<typename T>
T TVMArgs::At(int i) const {
    TVMArgValue arg = operator[](i);
    return arg.operator T();
}


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

/*!
 * \brief The name of DLDeviceType.
 * \param type The device type.
 * \return the device name.
 */
inline const char* DLDeviceType2Str(int type) {
    switch (type) {
        case static_cast<int>(DLDeviceType::kDLCPU):
            return "cpu";
        case static_cast<int>(DLDeviceType::kDLCUDA):
            return "cuda";
        case static_cast<int>(DLDeviceType::kDLCUDAHost):
            return "cuda_host";
        case static_cast<int>(DLDeviceType::kDLCUDAManaged):
            return "cuda_managed";
        case static_cast<int>(DLDeviceType::kDLOpenCL):
            return "opencl";
        case static_cast<int>(TVMDeviceExtType::kDLSDAccel):
            return "sdaccel";
        case static_cast<int>(TVMDeviceExtType::kDLAOCL):
            return "aocl";
        case static_cast<int>(DLDeviceType::kDLVulkan):
            return "vulkan";
        case static_cast<int>(DLDeviceType::kDLMetal):
            return "metal";
        case static_cast<int>(DLDeviceType::kDLVPI):
            return "vpi";
        case static_cast<int>(DLDeviceType::kDLROCM):
            return "rocm";
        case static_cast<int>(DLDeviceType::kDLROCMHost):
            return "rocm_host";
        case static_cast<int>(DLDeviceType::kDLExtDev):
            return "ext_dev";
        case static_cast<int>(DLDeviceType::kDLOneAPI):
            return "oneapi";
        case static_cast<int>(DLDeviceType::kDLWebGPU):
            return "webgpu";
        case static_cast<int>(DLDeviceType::kDLHexagon):
            return "hexagon";
        case static_cast<int>(TVMDeviceExtType::kOpenGL):
            return "opengl";
        case static_cast<int>(TVMDeviceExtType::kDLMicroDev):
            return "microdev";
        default:
            LOG(FATAL) << "unknown type = " << type;
    }
    throw;
}

template<typename Derived>
template<typename TObjectRef, typename>
bool TVMPODValue_CRTP_<Derived>::IsObjectRef() const {
    using ContainerType = typename TObjectRef::ContainerType;
    // NOTE: the following code can be optimized by constant folding.
    if (std::is_base_of<NDArray::ContainerType, ContainerType>::value) {
        return type_code_ == static_cast<int>(TVMArgTypeCode::kTVMNDArrayHandle) &&
               TVMArrayHandleToObjectHandle(static_cast<TVMArrayHandle>(value_.v_handle))
                       ->IsInstance<ContainerType>();
    }

    if (std::is_base_of<Module::ContainerType, ContainerType>::value) {
        return type_code_ == static_cast<int>(TVMArgTypeCode::kTVMModuleHandle) &&
               static_cast<Object*>(value_.v_handle)->IsInstance<ContainerType>();
    }

    if (std::is_base_of<PackedFunc::ContainerType, ContainerType>::value) {
        return type_code_ == static_cast<int>(TVMArgTypeCode::kTVMPackedFuncHandle) &&
               static_cast<Object*>(value_.v_handle)->IsInstance<ContainerType>();
    }

    // NOTE: we don't pass NDArray and runtime::Module as RValue ref.
    if (type_code_ == static_cast<int>(TVMArgTypeCode::kTVMObjectRValueRefArg)) {
        return ObjectTypeChecker<TObjectRef>::Check(*static_cast<Object**>(value_.v_handle));
    }

    return (std::is_base_of<ContainerType, NDArray::ContainerType>::value &&
            type_code_ == static_cast<int>(TVMArgTypeCode::kTVMNDArrayHandle)) ||
           (std::is_base_of<ContainerType, Module::ContainerType>::value &&
            type_code_ == static_cast<int>(TVMArgTypeCode::kTVMModuleHandle)) ||
           (std::is_base_of<ContainerType, PackedFunc::ContainerType>::value &&
            type_code_ == static_cast<int>(TVMArgTypeCode::kTVMPackedFuncHandle)) ||
           (type_code_ == static_cast<int>(TVMArgTypeCode::kTVMObjectHandle) &&
            ObjectTypeChecker<TObjectRef>::Check(static_cast<Object*>(value_.v_handle)));
}

template<typename Derived>
template<typename TObjectRef>
TObjectRef TVMPODValue_CRTP_<Derived>::AsObjectRef() const {
    static_assert(std::is_base_of<ObjectRef, TObjectRef>::value,
                  "Conversion only works for ObjectRef");
    using ContainerType = typename TObjectRef::ContainerType;

    if (type_code_ == static_cast<int>(TVMArgTypeCode::kTVMNullptr)) {
        CHECK(TObjectRef::_type_is_nullable)
                << "Expect a not null value of " << ContainerType::_type_key;
        return TObjectRef(ObjectPtr<Object>(nullptr));
    }

    // NOTE: The following code uses "if constexpr" wherever possible to
    // minimize the number of runtime checks.
    if constexpr (std::is_base_of_v<NDArray::ContainerType, ContainerType>) {
        // Casting to a sub-class of NDArray
        TVM_CHECK_TYPE_CODE(type_code_, static_cast<int>(TVMArgTypeCode::kTVMNDArrayHandle));
        ObjectPtr<Object> data =
                NDArray::FFIDataFromHandle(static_cast<TVMArrayHandle>(value_.v_handle));
        CHECK(data->IsInstance<ContainerType>())
                << "Expected " << ContainerType::_type_key << " but got " << data->GetTypeKey();
        return TObjectRef(data);
    }

    if constexpr (std::is_base_of_v<Module::ContainerType, ContainerType>) {
        // Casting to a sub-class of Module
        TVM_CHECK_TYPE_CODE(type_code_, static_cast<int>(TVMArgTypeCode::kTVMModuleHandle));
        ObjectPtr<Object> data = GetObjectPtr<Object>(static_cast<Object*>(value_.v_handle));
        CHECK(data->IsInstance<ContainerType>())
                << "Expected " << ContainerType::_type_key << " but got " << data->GetTypeKey();
        return TObjectRef(data);
    }

    if constexpr (std::is_base_of_v<PackedFunc::ContainerType, ContainerType>) {
        // Casting to a sub-class of PackedFunc
        TVM_CHECK_TYPE_CODE(type_code_, static_cast<int>(TVMArgTypeCode::kTVMPackedFuncHandle));
        ObjectPtr<Object> data = GetObjectPtr<Object>(static_cast<Object*>(value_.v_handle));
        CHECK(data->IsInstance<ContainerType>())
                << "Expected " << ContainerType::_type_key << " but got " << data->GetTypeKey();
        return TObjectRef(data);
    }

    if (type_code_ == static_cast<int>(TVMArgTypeCode::kTVMObjectHandle)) {
        // normal object type check.
        auto* ptr = static_cast<Object*>(value_.v_handle);
        Optional<String> checked_type = ObjectTypeChecker<TObjectRef>::CheckAndGetMismatch(ptr);
        CHECK(!checked_type.defined()) << "Expected " << ObjectTypeChecker<TObjectRef>::TypeName()
                                       << ", but got " << checked_type.value();
        return TObjectRef(GetObjectPtr<Object>(ptr));
    }

    if (type_code_ == static_cast<int>(TVMArgTypeCode::kTVMObjectRValueRefArg)) {
        Object* ptr = *static_cast<Object**>(value_.v_handle);
        Optional<String> checked_type = ObjectTypeChecker<TObjectRef>::CheckAndGetMismatch(ptr);
        CHECK(!checked_type.defined()) << "Expected " << ObjectTypeChecker<TObjectRef>::TypeName()
                                       << ", but got " << checked_type.value();
        return TObjectRef(GetObjectPtr<Object>(ptr));
    }

    if constexpr (std::is_base_of_v<ContainerType, NDArray::ContainerType>) {
        if (type_code_ == static_cast<int>(TVMArgTypeCode::kTVMNDArrayHandle)) {
            // Casting to a base class that NDArray can sub-class
            ObjectPtr<Object> data =
                    NDArray::FFIDataFromHandle(static_cast<TVMArrayHandle>(value_.v_handle));
            return TObjectRef(data);
        }
    }

    if constexpr (std::is_base_of_v<ContainerType, Module::ContainerType>) {
        if (type_code_ == static_cast<int>(TVMArgTypeCode::kTVMModuleHandle)) {
            // Casting to a base class that Module can sub-class
            return TObjectRef(GetObjectPtr<Object>(static_cast<Object*>(value_.v_handle)));
        }
    }

    if constexpr (std::is_base_of_v<ContainerType, PackedFunc::ContainerType>) {
        if (type_code_ == static_cast<int>(TVMArgTypeCode::kTVMPackedFuncHandle)) {
            // Casting to a base class that PackedFunc can sub-class
            return TObjectRef(GetObjectPtr<Object>(static_cast<Object*>(value_.v_handle)));
        }
    }

    if constexpr (std::is_base_of_v<TObjectRef, Int>) {
        if (type_code_ == static_cast<int>(TVMArgTypeCode::kTVMArgInt)) {
            return Int(value_.v_int64);
        }
    }

    if constexpr (std::is_base_of_v<TObjectRef, Float>) {
        if (type_code_ == static_cast<int>(TVMArgTypeCode::kTVMArgFloat)) {
            return Float(value_.v_float64);
        }
    }

    if constexpr (std::is_base_of_v<TObjectRef, Bool>) {
        if (type_code_ == static_cast<int>(TVMArgTypeCode::kTVMArgBool)) {
            return Bool(value_.v_int64);
        }
    }

    if constexpr (std::is_base_of_v<TObjectRef, String>) {
        if (type_code_ == static_cast<int>(TVMArgTypeCode::kTVMStr) ||
            type_code_ == static_cast<int>(TVMArgTypeCode::kTVMBytes)) {
            // This step is the reason why `AsObjectRef` cannot be provided
            // in the base `TVMPODValue_` class.  Because `TVMArgValue` and
            // `TVMRetValue` have different implementations of `operator
            // std::string`, with different interpretations of `kTVMStr` and
            // `kTVMBytes`, we must delegate to those implementations.
            //
            // This could be done with a pure virtual method in
            // `TVMPODValue_`, but that would require a vtable lookup during
            // FFI conversions, imposing a runtime overhead.
            return String(static_cast<const Derived*>(this)->operator std::string());
        }
    }

    TVM_CHECK_TYPE_CODE(type_code_, static_cast<int>(TVMArgTypeCode::kTVMObjectHandle));
    return TObjectRef(ObjectPtr<Object>(nullptr));
}

inline TVMArgValue::operator DLDataType() const {
    if (String::CanConvertFrom(*this)) {
        return String2DLDataType(PackedFuncValueConverter<String>::From(*this).operator std::string());
    }
    // None type
    if (type_code_ == static_cast<int>(TVMArgTypeCode::kTVMNullptr)) {
        DLDataType t;
        t.code = static_cast<uint8_t>(TVMArgTypeCode::kTVMOpaqueHandle);
        t.bits = 0;
        t.lanes = 0;
        return t;
    }
    TVM_CHECK_TYPE_CODE(type_code_, static_cast<int>(TVMArgTypeCode::kTVMDataType));
    return value_.v_type;
}

inline TVMArgValue::operator DataType() const {
    return DataType(operator DLDataType());
}

}// namespace litetvm::runtime

#endif//PACKED_FUNC_H
