//
// Created by 赵丹 on 25-1-13.
//

#ifndef C_RUNTIME_API_H
#define C_RUNTIME_API_H

/*!
 * \brief The type code options DLDataType.
 */
enum class DLDataTypeCode : uint8_t {
    /*! \brief signed integer */
    kDLInt = 0U,
    /*! \brief unsigned integer */
    kDLUInt = 1U,
    /*! \brief IEEE floating point */
    kDLFloat = 2U,
    /*!
 * \brief Opaque handle type, reserved for testing purposes.
 * Frameworks need to agree on the handle data type for the exchange to be well-defined.
 */
    kDLOpaqueHandle = 3U,
    /*! \brief bfloat16 */
    kDLBfloat = 4U,
    /*!
 * \brief complex number
 * (C/C++/Python layout: compact struct per complex number)
 */
    kDLComplex = 5U,
};


/*!
 * \brief The type code in used and only used in TVM FFI for argument passing.
 *
 * DLPack consistency:
 * 1) kTVMArgInt is compatible with kDLInt
 * 2) kTVMArgFloat is compatible with kDLFloat
 * 3) kDLUInt is not in ArgTypeCode, but has a spared slot
 *
 * Downstream consistency:
 * The kDLInt, kDLUInt, kDLFloat are kept consistent with the original ArgType code
 *
 * It is only used in argument passing, and should not be confused with
 * DataType::TypeCode, which is DLPack-compatible.
 *
 * \sa tvm::runtime::DataType::TypeCode
 */
enum class TVMArgTypeCode : uint8_t {
    kTVMArgInt = static_cast<uint8_t>(DLDataTypeCode::kDLInt),
    kTVMArgFloat = static_cast<uint8_t>(DLDataTypeCode::kDLFloat),
    kTVMOpaqueHandle = 3U,
    kTVMNullptr = 4U,
    kTVMDataType = 5U,
    kDLDevice = 6U,
    kTVMDLTensorHandle = 7U,
    kTVMObjectHandle = 8U,
    kTVMModuleHandle = 9U,
    kTVMPackedFuncHandle = 10U,
    kTVMStr = 11U,
    kTVMBytes = 12U,
    kTVMNDArrayHandle = 13U,
    kTVMObjectRValueRefArg = 14U,
    kTVMArgBool = 15U,
    // Extension codes for other frameworks to integrate TVM PackedFunc.
    // To make sure each framework's id do not conflict, use first and
    // last sections to mark ranges.
    // Open an issue at the repo if you need a section of code.
    kTVMExtBegin = 16U,
    kTVMNNVMFirst = 16U,
    kTVMNNVMLast = 20U,
    // The following section of code is used for non-reserved types.
    kTVMExtReserveEnd = 64U,
    kTVMExtEnd = 128U,
};

#endif//C_RUNTIME_API_H
