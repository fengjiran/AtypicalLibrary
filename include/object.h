//
// Created by richard on 7/3/24.
//

#ifndef ATYPICALLIBRARY_OBJECT_H
#define ATYPICALLIBRARY_OBJECT_H

#include <cstdint>

// nested namespace
namespace litetvm::runtime {

/**
 * @brief Namespace for the list of type index
 */
enum class TypeIndex {
    /// \brief Root object type.
    kRoot = 0,

    /// \brief runtime::Module.
    /// <p>
    /// Standard static index assignment,
    /// frontend can take benefit of these constants.
    kRuntimeModule = 1,

    /// \brief runtime::NDArray.
    kRuntimeNDArray = 2,

    /// \brief runtime::String
    kRuntimeString = 3,

    /// \brief runtime::Array.
    kRuntimeArray = 4,

    /// \brief runtime::Map.
    kRuntimeMap = 5,

    /// \brief runtime::ShapeTuple.
    kRuntimeShapeTuple = 6,

    /// \brief runtime::PackedFunc.
    kRuntimePackedFunc = 7,

    /// \brief runtime::DRef for disco distributed runtime.
    kRuntimeDiscoDRef = 8,

    /// \brief runtime::RPCObjectRef.
    kRuntimeRPCObjectRef = 9,

    /// \brief static assignments that may subject to change.
    kRuntimeClosure,
    kRuntimeADT,
    kStaticIndexEnd,

    /// \brief Type index is allocated during runtime.
    kDynamic = kStaticIndexEnd
};

}// namespace litetvm::runtime


#endif//ATYPICALLIBRARY_OBJECT_H
