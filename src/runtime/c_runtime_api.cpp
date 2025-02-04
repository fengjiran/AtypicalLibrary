//
// Created by richard on 2/4/25.
//

#include "runtime/c_runtime_api.h"
#include "runtime/device_api.h"
#include "runtime/packed_func.h"

namespace litetvm::runtime {



uint8_t ParseCustomDatatype(const std::string& s, const char** scan) {
    CHECK(s.substr(0, 6) == "custom") << "Not a valid custom datatype string";

    auto tmp = s.c_str();

    CHECK(s.c_str() == tmp);
    *scan = s.c_str() + 6;
    CHECK(s.c_str() == tmp);
    if (**scan != '[') LOG(FATAL) << "expected opening brace after 'custom' type in" << s;
    CHECK(s.c_str() == tmp);
    *scan += 1;
    CHECK(s.c_str() == tmp);
    size_t custom_name_len = 0;
    CHECK(s.c_str() == tmp);
    while (*scan + custom_name_len <= s.c_str() + s.length() && *(*scan + custom_name_len) != ']')
        ++custom_name_len;
    CHECK(s.c_str() == tmp);
    if (*(*scan + custom_name_len) != ']')
        LOG(FATAL) << "expected closing brace after 'custom' type in" << s;
    CHECK(s.c_str() == tmp);
    *scan += custom_name_len + 1;
    CHECK(s.c_str() == tmp);

    auto type_name = s.substr(7, custom_name_len);
    CHECK(s.c_str() == tmp);
    return GetCustomTypeCode(type_name);
}

}