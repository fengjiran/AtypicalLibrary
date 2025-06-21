//
// Created by 赵丹 on 25-6-19.
//
#include "tensor.h"

#include <fmt/compile.h>
#include <fmt/format.h>
#include <fmt/ostream.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <string>
#include <tuple>

namespace atp {

enum class FormatType {
    Default,   // 'g' format (defaultfloat equivalent)
    Scientific,// 'e' format with precision 4
    Fixed      // 'f' format with precision 4
};

struct PrintFormat {
    double scale;
    int width;
    FormatType type;

    PrintFormat(double s, int w, FormatType t = FormatType::Default)
        : scale(s), width(w), type(t) {}
};

static PrintFormat print_format(const Tensor& t) {
    using data_type = DataType2CPPType<DLDataTypeCode::kFloat, 32, 1>::type;
    auto size = t.numel();
    if (size == 0) {
        return {1.0, 0};
    }

    bool int_mod = true;
    // auto data = t.const_data_ptr();
    for (int64_t i = 0; i < size; ++i) {
        // auto z =
    }
}

}// namespace atp