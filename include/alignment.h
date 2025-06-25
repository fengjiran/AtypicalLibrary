//
// Created by richard on 6/25/25.
//

#ifndef ALIGNMENT_H
#define ALIGNMENT_H

namespace atp {

constexpr size_t gAlignment = 64;

constexpr size_t gPagesize = 4096;
// since the default thp pagesize is 2MB, enable thp only
// for buffers of size 2MB or larger to avoid memory bloating
constexpr size_t gAlloc_threshold_thp = static_cast<size_t>(2) * 1024 * 1024;
}

#endif //ALIGNMENT_H
