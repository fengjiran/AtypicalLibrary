//
// Created by 赵丹 on 24-5-18.
//

#ifndef ATYPICALLIBRARY_CONFIG_HPP
#define ATYPICALLIBRARY_CONFIG_HPP

#ifdef __cplusplus

// cpp version check
#ifndef CPP_STD_VERSION
#if __cplusplus <= 201103L
#define CPP_STD_VERSION 11
#elif __cplusplus <= 201402L
#define CPP_STD_VERSION 14
#elif __cplusplus <= 201703L
#define CPP_STD_VERSION 17
#elif __cplusplus <= 202002L
#define CPP_STD_VERSION 20
#elif __cplusplus <= 202302L
#define CPP_STD_VERSION 23
#else
#define CPP_STD_VERSION 26
#endif
#endif// CPP_STD_VERSION

#if __has_cpp_attribute(nodiscard)
#define CPP_NODISCARD [[nodiscard]]
#else
#define CPP_NODISCARD
#endif

#endif

#endif//ATYPICALLIBRARY_CONFIG_HPP
