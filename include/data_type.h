//
// Created by 赵丹 on 25-1-13.
//

#ifndef DATA_TYPE_H
#define DATA_TYPE_H

#include "c_runtime_api.h"
#include "glog/logging.h"

namespace litetvm::runtime {

/*!
 * \brief The data type the tensor can hold. The data type is assumed to follow the
 * native endian-ness. An explicit error message should be raised when attempting to
 * export an array with non-native endianness
 *
 *  Examples
 *   - float: type_code = 2, bits = 32, lanes=1
 *   - float4(vectorized 4 float): type_code = 2, bits = 32, lanes=4
 *   - int8: type_code = 0, bits = 8, lanes=1
 *   - std::complex<float>: type_code = 5, bits = 64, lanes = 1
 */
struct DLDataType {
    /*!
   * \brief Type code of base types.
   * We keep it uint8_t instead of DLDataTypeCode for minimal memory
   * footprint, but the value should be one of DLDataTypeCode enum values.
   * */
    uint8_t code;
    /*!
   * \brief Number of bits, common choices are 8, 16, 32.
   */
    uint8_t bits;
    /*! \brief Number of lanes in the type, used for vector types. */
    uint16_t lanes;
};

/*!
 * \brief Runtime primitive data type.
 *
 *  This class is a thin wrapper of DLDataType.
 *  We also make use of DataType in compiler to store quick hint
 */
class DataType {
public:
    /*!
     * \brief Type code for the DataType.
     *
     * DLPack consistency:
     * 1) kInt is consistent with kDLInt
     * 2) kUInt is consistent with kDLUInt
     * 3) kFloat is consistent with kDLFloat
     */
    enum class TypeCode : uint8_t {
        kInt = static_cast<uint8_t>(DLDataTypeCode::kDLInt),
        kUInt = static_cast<uint8_t>(DLDataTypeCode::kDLUInt),
        kFloat = static_cast<uint8_t>(DLDataTypeCode::kDLFloat),
        kHandle = static_cast<uint8_t>(TVMArgTypeCode::kTVMOpaqueHandle),
        kBFloat = static_cast<uint8_t>(DLDataTypeCode::kDLBfloat),
        kE4M3Float = 6U,
        kE5M2Float = 7U,
        kCustomBegin = 129
    };

    /*!
   * \brief Constructor
   * \param dtype The DLDataType
   */
    explicit DataType(DLDataType dtype) : dtype_(dtype) {}

    /*!
   * \brief Constructor
   * \param code The type code.
   * \param bits The number of bits in the type.
   * \param lanes The number of lanes.
   * \param is_scalable Whether the data type is scalable.
   */
    DataType(int code, int bits, int lanes, bool is_scalable = false) {
        dtype_.code = static_cast<uint8_t>(code);
        dtype_.bits = static_cast<uint8_t>(bits);
        if (is_scalable) {
            CHECK(lanes > 1) << "Invalid value for vscale factor" << lanes;
        }
    }

    /*!
   * \brief Construct a Void type.
   * \return The constructed data type.
   */
    static DataType Void() {
        return {static_cast<int>(TypeCode::kHandle), 0, 0};
    }


private:
    DLDataType dtype_;
};

}// namespace litetvm::runtime


#endif//DATA_TYPE_H
