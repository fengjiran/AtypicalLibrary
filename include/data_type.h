//
// Created by 赵丹 on 25-1-13.
//

#ifndef DATA_TYPE_H
#define DATA_TYPE_H

#include "c_runtime_api.h"
#include "glog/logging.h"

#include <c_runtime_api.h>

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

    /*! \brief default constructor */
    DataType() : dtype_(Void()) {}

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
     * \param is_scalable Whether the data type is scalable (dynamic array)
     */
    DataType(int code, int bits, int lanes, bool is_scalable = false) {
        dtype_.code = static_cast<uint8_t>(code);
        dtype_.bits = static_cast<uint8_t>(bits);
        if (is_scalable) {
            CHECK(lanes > 1) << "Invalid value for vscale factor: " << lanes;
        }
        dtype_.lanes = is_scalable ? static_cast<uint16_t>(-lanes) : static_cast<uint16_t>(lanes);

        if (code == static_cast<int>(TypeCode::kFloat)) {
            CHECK_EQ(bits, 16);
        }

        if (code == static_cast<int>(TypeCode::kE4M3Float) || code == static_cast<int>(TypeCode::kE5M2Float)) {
            CHECK_EQ(bits, 8);
        }
    }

    /*! \return The type code. */
    int code() const {
        return dtype_.code;
    }

    /*! \return number of bits in the data. */
    int bits() const {
        return dtype_.bits;
    }

    /*! \return number of bytes to store each scalar. */
    int bytes() const {
        return (dtype_.bits + 7) / 8;
    }

    /*! \return number of lanes in the data. */
    int lanes() const {
        int lanes = static_cast<int16_t>(dtype_.lanes);
        if (lanes < 0) {
            LOG(FATAL) << "Can't fetch the lanes of a scalable vector at a compile time.";
        }
        return lanes;
    }

    /*! \return the integer multiplier of vscale in a scalable vector. */
    int vscale_factor() const {
        int lanes = static_cast<int16_t>(dtype_.lanes);
        if (lanes >= -1) {
            LOG(FATAL) << "A fixed length vector doesn't have a vscale factor.";
        }
        return -lanes;
    }

    /*! \return get vscale factor or lanes depending on scalability of the vector. */
    int get_lanes_or_vscale_factor() const {
        return is_scalable_vector() ? vscale_factor() : lanes();
    }

    /*! \return whether type is a scalar type. */
    bool is_scalar() const {
        return !is_scalable_vector() && lanes() == 1;
    }
    /*! \return whether type is a scalar type. */
    bool is_bool() const {
        return code() == static_cast<int>(TypeCode::kUInt) && bits() == 1;
    }
    /*! \return whether type is a float type. */
    bool is_float() const {
        return code() == static_cast<int>(TypeCode::kFloat);
    }
    /*! \return whether type is a float8 type. */
    bool is_float8() const {
        return (code() == static_cast<int>(TypeCode::kFloat) ||
                code() == static_cast<int>(TypeCode::kE4M3Float) ||
                code() == static_cast<int>(TypeCode::kE5M2Float)) &&
               bits() == 8;
    }

    bool is_e4m3_float8() const {
        return code() == static_cast<int>(TypeCode::kE4M3Float) && bits() == 8;
    }

    bool is_e5m2_float8() const {
        return code() == static_cast<int>(TypeCode::kE5M2Float) && bits() == 8;
    }

    /*! \return whether type is a float16 type. */
    bool is_float16() const {
        return is_float() && bits() == 16;
    }
    /*! \return whether type is a bfloat16 type. */
    bool is_bfloat16() const {
        return code() == static_cast<int>(TypeCode::kBFloat) && bits() == 16;
    }

    /*! \return whether type is an int type. */
    bool is_int() const {
        return code() == (int) TypeCode::kInt;
    }

    /*! \return whether type is an uint type. */
    bool is_uint() const {
        return code() == static_cast<int>(TypeCode::kUInt);
    }

    /*! \return whether type is a handle type. */
    bool is_handle() const {
        return code() == static_cast<int>(TypeCode::kHandle) && !is_void();
    }

    /*! \return whether type is a vector type. */
    bool is_scalable_or_fixed_length_vector() const {
        int encoded_lanes = static_cast<int16_t>(dtype_.lanes);
        return encoded_lanes < -1 || 1 < encoded_lanes;
    }

    /*! \return Whether the type is a scalable vector. */
    bool is_scalable_vector() const {
        return static_cast<int16_t>(dtype_.lanes) < -1;
    }

    /*! \return whether type is a vector type. */
    bool is_vector() const {
        return lanes() > 1;
    }

    /*! \return whether type is a bool vector type. */
    bool is_vector_bool() const {
        return is_scalable_or_fixed_length_vector() && bits() == 1;
    }

    /*! \return whether type is a Void type. */
    bool is_void() const {
        return code() == static_cast<int>(TypeCode::kHandle) && bits() == 0 && lanes() == 0;
    }

    /*!
   * \brief Create a new data type by change lanes to a specified value.
   * \param lanes The target number of lanes.
   * \return the result type.
   */
    DataType with_lanes(int lanes) const {
        return {code(), bits(), lanes};
    }

    /*!
   * \brief Create a new scalable vector data type by changing the vscale multiplier to a specified
   * value. We'll use the data_.lanes field for this value.
   * \param vscale_factor The vscale multiplier.
   * \return A copy of the old DataType with the number of scalable lanes.
   */
    DataType with_scalable_vscale_factor(int vscale_factor) const {
        return {code(), bits(), -vscale_factor};
    }

    /*!
   * \brief Create a new data type by change bits to a specified value.
   * \param bits The target number of bits.
   * \return the result type.
   */
    DataType with_bits(int bits) const {
        return {code(), bits, dtype_.lanes};
    }

    /*!
   * \brief Get the scalar version of the type.
   * \return the result type.
   */
    DataType element_of() const {
        return with_lanes(1);
    }

    /*!
   * \brief Assignment operator.
   */
    DataType& operator=(const DataType& rhs) {
        DataType tmp(rhs);
        std::swap(tmp, *this);
        return *this;
    }

    /*!
   * \brief Equal comparator.
   * \param other The data type to compare against.
   * \return The comparison result.
   */
    bool operator==(const DataType& other) const {
        return dtype_.code == other.dtype_.code &&
               dtype_.bits == other.dtype_.bits &&
               dtype_.lanes == other.dtype_.lanes;
    }

    /*!
   * \brief NotEqual comparator.
   * \param other The data type to compare against.
   * \return The comparison result.
   */
    bool operator!=(const DataType& other) const {
        // return !operator==(other);
        return !(*this == other);
    }

    /*!
     * \brief Construct a Void type.
     * \return The constructed data type.
     */
    static DataType Void() {
        return {static_cast<uint8_t>(TypeCode::kHandle), 0, 0};
    }

    /*!
     * \brief Converter to DLDataType
     * \return the result.
     */
    explicit operator DLDataType() const {
        return dtype_;
    }

    /*!
   * \brief Construct an int type.
   * \param bits The number of bits in the type.
   * \param lanes The number of lanes.
   * \return The constructed data type.
   */
    static DataType Int(int bits, int lanes = 1) {
        return {static_cast<uint8_t>(TypeCode::kInt), bits, lanes};
    }

    /*!
   * \brief Construct an uint type.
   * \param bits The number of bits in the type.
   * \param lanes The number of lanes.
   * \param is_scalable Whether the data type is scalable.
   * \return The constructed data type.
   */
    static DataType UInt(int bits, int lanes = 1, bool is_scalable = false) {
        return {static_cast<uint8_t>(TypeCode::kUInt), bits, lanes, is_scalable};
    }

    /*!
   * \brief Construct an float type.
   * \param bits The number of bits in the type.
   * \param lanes The number of lanes
   * \return The constructed data type.
   */
    static DataType Float(int bits, int lanes = 1) {
        return {static_cast<uint8_t>(TypeCode::kFloat), bits, lanes};
    }

    /*!
   * \brief Construct an bfloat type.
   * \param bits The number of bits in the type.
   * \param lanes The number of lanes
   * \return The constructed data type.
   */
    static DataType BFloat(int bits, int lanes = 1) {
        return {static_cast<uint8_t>(TypeCode::kBFloat), bits, lanes};
    }

    /*!
   * \brief Construct NV float8 e4m3 datatype.
   * \param lanes The number of lanes
   * \return The constructed data type.
   */
    static DataType NVFloat8E4M3(int lanes = 1) {
        return {static_cast<uint8_t>(TypeCode::kE4M3Float), 8, lanes};
    }

    /*!
   * \brief Construct NV float8 e5m2 datatype.
   * \param lanes The number of lanes
   * \return The constructed data type.
   */
    static DataType NVFloat8E5M2(int lanes = 1) {
        return {static_cast<uint8_t>(TypeCode::kE5M2Float), 8, lanes};
    }

    /*!
   * \brief Construct a bool type.
   * \param lanes The number of lanes.
   * \param is_scalable Whether the data type is scalable.
   * \return The constructed data type.
   */
    static DataType Bool(int lanes = 1, bool is_scalable = false) {
        return UInt(1, lanes, is_scalable);
    }

    /*!
   * \brief Construct a handle type.
   * \param bits The number of bits in the type.
   * \param lanes The number of lanes
   * \return The constructed data type.
   */
    static DataType Handle(int bits = 64, int lanes = 1) {
        return {static_cast<uint8_t>(TypeCode::kHandle), bits, lanes};
    }

    /*!
   * \brief Get the corresponding type of TVMShapeIndex.
   * \return The type of TVM shape index.
   */
    static DataType ShapeIndex() {
        if (std::is_signed_v<tvm_index_t>) {
            return Int(sizeof(tvm_index_t) * 8);
        } else {
            return UInt(sizeof(tvm_index_t) * 8);
        }
    }

private:
    DLDataType dtype_{};
};

/*!
 * \brief Get the number of bytes needed in a vector.
 * \param dtype The data type.
 * \return Number of bytes needed.
 */
inline int GetVectorBytes(DataType dtype) {
    if (dtype == DataType::Bool() ||
        dtype == DataType::Int(1) ||
        dtype == DataType::Int(4) ||
        dtype == DataType::UInt(4)) {
        return 1;
    }

    int bits = dtype.bits() * dtype.lanes();
    CHECK_EQ(bits % 8, 0U) << "Need to load/store by multiple of bytes";
    return bits / 8;
}

/*!
 * \brief Check whether type matches the given spec.
 * \param t The type
 * \param code The type code.
 * \param bits The number of bits to be matched.
 * \param lanes The number of lanes in the type.
 */
inline bool TypeMatch(DLDataType t, int code, int bits, int lanes = 1) {
    return t.code == code && t.bits == bits && t.lanes == lanes;
}

/*!
 * \brief Check whether two types are equal .
 * \param lhs The left operand.
 * \param rhs The right operand.
 */
inline bool TypeEqual(DLDataType lhs, DLDataType rhs) {
    return lhs.code == rhs.code && lhs.bits == rhs.bits && lhs.lanes == rhs.lanes;
}

}// namespace litetvm::runtime


#endif//DATA_TYPE_H
