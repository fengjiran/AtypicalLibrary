//
// Created by richard on 6/30/25.
//

#ifndef DATA_TYPE_H
#define DATA_TYPE_H

#include <cstdint>
#include <string>

namespace atp {

enum class DLDataTypeCode : uint8_t {
    kInt = 0,
    kUInt = 1,
    kBool,
    kFloat,
    kBfloat,
    kFloat8_e3m4,
    kFloat8_e4m3,
    kFloat8_e4m3b11fnuz,
    kFloat8_e4m3fn,
    kFloat8_e4m3fnuz,
    kFloat8_e5m2,
    kFloat8_e5m2fnuz,
    kFloat8_e8m0fnu,
    kFloat6_e2m3fn,
    kFloat6_e3m2fn,
    kFloat4_e2m1fn,
    Undefined,
};

/*!
 * \brief The data type the tensor can hold. The data type is assumed to follow the
 * native endian-ness. An explicit error message should be raised when attempting to
 * export an array with non-native endianness
 *
 *  Examples
 *   - float: type_code = 2, bits = 32, lanes = 1
 *   - float4(vectorized 4 float): type_code = 2, bits = 32, lanes = 4
 *   - int8: type_code = 0, bits = 8, lanes = 1
 *   - std::complex<float>: type_code = 5, bits = 64, lanes = 1
 *   - bool: type_code = 6, bits = 8, lanes = 1 (as per common array library convention, the underlying storage size of bool is 8 bits)
 *   - float8_e4m3: type_code = 8, bits = 8, lanes = 1 (packed in memory)
 *   - float6_e3m2fn: type_code = 16, bits = 6, lanes = 1 (packed in memory)
 *   - float4_e2m1fn: type_code = 17, bits = 4, lanes = 1 (packed in memory)
 *
 *  When a sub-byte type is packed, DLPack requires the data to be in little bit-endian, i.e.,
 *  for a packed data set D ((D >> (i * bits)) && bit_mask) stores the i-th element.
 */
struct DLDataType {
    // Data type code.
    DLDataTypeCode code;

    // Number of bits per data element.
    // For example, 8 for int8_t, 32 for float.
    int8_t bits;

    // Number of lanes per data element, for vector.
    int16_t lanes;
};

inline std::string Type2Str(const DLDataType& dtype) {
    switch (dtype.code) {
        case DLDataTypeCode::kInt: {
            return "int";
        }

        case DLDataTypeCode::kUInt: {
            return "uint";
        }

        case DLDataTypeCode::kFloat: {
            return "float";
        }

        case DLDataTypeCode::kBfloat: {
            return "bfloat";
        }

        case DLDataTypeCode::kFloat8_e3m4: {
            return "float8_e3m4";
        }

        case DLDataTypeCode::kFloat8_e4m3: {
            return "float8_e4m3";
        }

        case DLDataTypeCode::kFloat8_e4m3b11fnuz: {
            return "float8_e4m3b11fnuz";
        }

        case DLDataTypeCode::kFloat8_e4m3fn: {
            return "float8_e4m3fn";
        }

        case DLDataTypeCode::kFloat8_e4m3fnuz: {
            return "float8_e4m3fnuz";
        }

        case DLDataTypeCode::kFloat8_e5m2: {
            return "float8_e5m2";
        }

        case DLDataTypeCode::kFloat8_e5m2fnuz: {
            return "float8_e5m2fnuz";
        }

        case DLDataTypeCode::kFloat8_e8m0fnu: {
            return "float8_e8m0fnu";
        }

        case DLDataTypeCode::kFloat6_e2m3fn: {
            return "float6_e2m3fn";
        }

        case DLDataTypeCode::kFloat6_e3m2fn: {
            return "float6_e3m2fn";
        }

        case DLDataTypeCode::kFloat4_e2m1fn: {
            return "float4_e2m1fn";
        }

        default: {
            throw std::runtime_error("Unsupported data type");
        }
    }
}

#define SCALAR_TYPES_NAME(f) \
    f(bool, Bool);           \
    f(uint8_t, Byte);        \
    f(int8_t, Char);         \
    f(uint16_t, UShort);     \
    f(int16_t, Short);       \
    f(uint32_t, UInt);       \
    f(int, Int);             \
    f(uint64_t, ULong);      \
    f(int64_t, Long);        \
    f(float, Float);         \
    f(double, Double);

// TODO: bfloat, float8, float6 and float4 type need to be defined.
#define SCALAR_TYPES_TO_CPP_TYPES(f)           \
    f(DLDataTypeCode::kInt, 8, 1, int8_t);     \
    f(DLDataTypeCode::kInt, 16, 1, int16_t);   \
    f(DLDataTypeCode::kInt, 32, 1, int32_t);   \
    f(DLDataTypeCode::kInt, 64, 1, int64_t);   \
    f(DLDataTypeCode::kUInt, 8, 1, uint8_t);   \
    f(DLDataTypeCode::kUInt, 16, 1, uint16_t); \
    f(DLDataTypeCode::kUInt, 32, 1, uint32_t); \
    f(DLDataTypeCode::kUInt, 64, 1, uint64_t); \
    f(DLDataTypeCode::kBool, 8, 1, bool);      \
    f(DLDataTypeCode::kFloat, 32, 1, float);   \
    f(DLDataTypeCode::kFloat, 64, 1, double);  \
    // f(DLDataTypeCode::kBfloat, 16, 1, float);            \
// f(DLDataTypeCode::kFloat8_e3m4, 8, 1, float);        \
// f(DLDataTypeCode::kFloat8_e4m3, 8, 1, float);        \
// f(DLDataTypeCode::kFloat8_e4m3b11fnuz, 8, 1, float); \
// f(DLDataTypeCode::kFloat8_e4m3fn, 8, 1, float);      \
// f(DLDataTypeCode::kFloat8_e4m3fnuz, 8, 1, float);    \
// f(DLDataTypeCode::kFloat8_e5m2, 8, 1, float);        \
// f(DLDataTypeCode::kFloat8_e5m2fnuz, 8, 1, float);    \
// f(DLDataTypeCode::kFloat8_e8m0fnu, 8, 1, float);     \
// f(DLDataTypeCode::kFloat6_e2m3fn, 6, 1, float);      \
// f(DLDataTypeCode::kFloat6_e3m2fn, 6, 1, float);      \
// f(DLDataTypeCode::kFloat4_e2m1fn, 4, 1, float)

}// namespace atp

#endif//DATA_TYPE_H
