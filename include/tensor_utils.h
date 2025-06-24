//
// Created by 赵丹 on 25-6-24.
//

#ifndef TENSOR_UTILS_H
#define TENSOR_UTILS_H

#ifdef __has_cpp_attribute
#if __has_cpp_attribute(nodiscard)
#define NODISCARD [[nodiscard]]
#else
#define NODISCARD
#endif

#if __has_cpp_attribute(maybe_unused)
#define MAYBE_UNUSED [[maybe_unused]]
#else
#define MAYBE_UNUSED
#endif
#endif

#if defined(__clang__)
#define __ubsan_ignore_pointer_overflow__ __attribute__((no_sanitize("pointer-overflow")))
#else
#define __ubsan_ignore_pointer_overflow__
#endif


#define UNUSED(expr)   \
    do {               \
        (void) (expr); \
    } while (false)

#endif//TENSOR_UTILS_H
