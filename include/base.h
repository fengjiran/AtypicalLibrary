//
// Created by 赵丹 on 24-12-31.
//

#ifndef ATYPICALLIBRARY_BASE_H
#define ATYPICALLIBRARY_BASE_H

#include <cstdint>
#include <glog/logging.h>
#include <utility>

#define UNUSED(expr)   \
    do {               \
        (void) (expr); \
    } while (0);

namespace base {

enum class StatusCode : uint8_t {
    kSuccess = 0,
    kFunctionUnImplement = 1,
    kPathNotValid = 2,
    kModelParseError = 3,
    kInternalError = 5,
    kKeyValueHasExist = 6,
    kInvalidArgument = 7,
};

enum class DeviceType : uint8_t {
    kDeviceUnknown = 0,
    kDeviceCPU = 1,
    kDeviceCUDA = 2,
    kDeviceCANN = 3,
};

enum class DataType : uint8_t {
    kDataTypeUnknown = 0,
    kDataTypeFp32 = 1,
    kDataTypeInt8 = 2,
    kDataTypeInt32 = 3,
};

enum class ModelType : uint8_t {
    kModelTypeUnknown = 0,
    kModelTypeLLama2 = 1,
};

enum class TokenizerType : int8_t {
    kEncodeUnknown = -1,
    kEncodeSpe = 0,
    kEncodeBpe = 1,
};

inline size_t DataTypeSize(DataType data_type) {
    if (data_type == DataType::kDataTypeFp32) {
        return sizeof(float);
    }

    if (data_type == DataType::kDataTypeInt8) {
        return sizeof(int8_t);
    }

    if (data_type == DataType::kDataTypeInt32) {
        return sizeof(int32_t);
    }

    return 0;
}

class Status {
public:
    Status() = default;
    Status(StatusCode code, std::string message) : code_(code), message_(std::move(message)) {}

    Status(const Status&) = default;
    Status& operator=(const Status&) = default;


private:
    StatusCode code_{StatusCode::kSuccess};
    std::string message_;
};


}// namespace base

#endif//ATYPICALLIBRARY_BASE_H
