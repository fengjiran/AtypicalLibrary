//
// Created by 赵丹 on 25-1-23.
//

#include "allocator.h"
#include "alignment.h"

#include <cstring>

#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace atp {

namespace {

// Fill the data memory region of num bytes with a particular garbage pattern.
// The garbage value is chosen to be NaN if interpreted as floating point value
// or a very large integer.
void memset_junk(void* data, size_t num) {
    // This garbage pattern is NaN when interpreted as floating point values
    // or as very large integer values.
    static constexpr int32_t kJunkPattern = 0x7fedbeef;
    static constexpr int64_t kJunkPattern64 = static_cast<int64_t>(kJunkPattern) << 32 | kJunkPattern;
    auto int64_count = num / sizeof(kJunkPattern64);
    auto remaining_bytes = num % sizeof(kJunkPattern64);
    auto* data_i64 = static_cast<int64_t*>(data);
    for (size_t i = 0; i < int64_count; ++i) {
        data_i64[i] = kJunkPattern64;
    }

    if (remaining_bytes > 0) {
        memcpy(data_i64 + int64_count, &kJunkPattern64, remaining_bytes);
    }
}

bool is_thp_alloc_enabled() {
    static bool value = [&] {
        auto env = check_env("THP_MEM_ALLOC_ENABLE");
        return env.has_value() ? env.value() : 0;
    }();
    return value;
}

bool is_thp_alloc(size_t nbytes) {
    return is_thp_alloc_enabled() && nbytes >= gAlloc_threshold_thp;
}

}// namespace


}// namespace atp
