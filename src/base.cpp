//
// Created by 赵丹 on 24-12-31.
//

#include "base.h"

namespace base {

namespace error {

Status Success(const std::string& err_msg) {
    return {StatusCode::kSuccess, err_msg};
}

Status FunctionNotImplement(const std::string& err_msg) {
    return {StatusCode::kFunctionUnImplement, err_msg};
}

Status PathNotValid(const std::string& err_msg) {
    return {StatusCode::kPathNotValid, err_msg};
}

Status ModelParseError(const std::string& err_msg) {
    return {StatusCode::kModelParseError, err_msg};
}

Status InternalError(const std::string& err_msg) {
    return {StatusCode::kInternalError, err_msg};
}

Status KeyHasExits(const std::string& err_msg) {
    return {StatusCode::kKeyValueHasExist, err_msg};
}

Status InvalidArgument(const std::string& err_msg) {
    return {StatusCode::kInvalidArgument, err_msg};
}

}// namespace error

std::ostream& operator<<(std::ostream& os, const Status& err) {
    os << err.get_err_message();
    return os;
}

}// namespace base