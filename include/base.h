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

#define CONCAT_(a, b) a## b
#define CONCAT(a, b) CONCAT_(a, b)
#define UNIQUE_ID(prefix) CONCAT(prefix, __COUNTER__)

#define MAX_(t1, t2, max1, max2, x, y) ({ \
    t1 max1 = (x);                        \
    t2 max2 = (y);                        \
    (void) (&max1 == &max2);              \
    max1 > max2 ? max1 : max2;            \
})
#define MAX(x, y) MAX_(decltype(x), decltype(y), UNIQUE_ID(max1_), UNIQUE_ID(max2_), x, y)

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
    Status(StatusCode code, std::string message)
        : code_(static_cast<int>(code)), message_(std::move(message)) {}

    Status(const Status&) = default;
    Status& operator=(const Status&) = default;
    Status& operator=(StatusCode code) {
        code_ = static_cast<int>(code);
        return *this;
    }

    Status& operator=(int code) {
        code_ = code;
        return *this;
    }

    bool operator==(StatusCode code) const {
        return code_ == static_cast<int>(code);
    }

    bool operator!=(StatusCode code) const {
        return code_ != static_cast<int>(code);
    }

    bool operator==(int code) const {
        return code_ == code;
    }

    bool operator!=(int code) const {
        return code_ != code;
    }

    operator int() const {
        return code_;
    }

    operator bool() const {
        return code_ == static_cast<int>(StatusCode::kSuccess);
    }

    int get_err_code() const {
        return code_;
    }

    const std::string& get_err_message() const {
        return message_;
    }

    void set_err_message(const std::string& message) {
        message_ = message;
    }

private:
    int code_{static_cast<int>(StatusCode::kSuccess)};
    std::string message_;
};

namespace error {

#define STATUS_CHECK(call)                                                             \
    do {                                                                               \
        const base::Status& status = call;                                             \
        if (!status) {                                                                 \
            char buf[512];                                                             \
            snprintf(buf, buf_size - 1,                                                \
                     "Infer error\n File:%s Line:%d\n Error code:%d\n Error msg:%s\n", \
                     __FILE__, __LINE__, int(status), status.get_err_msg().c_str());   \
            LOG(FATAL) << buf;                                                         \
        }                                                                              \
    } while (0);

Status Success(const std::string& err_msg = "");
Status FunctionNotImplement(const std::string& err_msg = "");
Status PathNotValid(const std::string& err_msg = "");
Status ModelParseError(const std::string& err_msg = "");
Status InternalError(const std::string& err_msg = "");
Status KeyHasExits(const std::string& err_msg = "");
Status InvalidArgument(const std::string& err_msg = "");

}// namespace error

std::ostream& operator<<(std::ostream& os, const Status& status);

}// namespace base

#endif//ATYPICALLIBRARY_BASE_H
