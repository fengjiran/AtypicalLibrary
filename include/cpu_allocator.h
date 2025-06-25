//
// Created by 赵丹 on 25-6-25.
//

#ifndef CPU_ALLOCATOR_H
#define CPU_ALLOCATOR_H

#include "allocator.h"

namespace atp {

void* alloc_cpu(size_t nbytes);

void free_cpu(void* data);

class CPUAllocator : public Allocator {
public:
    CPUAllocator() = default;

    NODISCARD void* allocate(size_t n) const override {
        // return malloc(n);
        return alloc_cpu(n);
    }

    void deallocate(void* p) const override {
        // free(p);
        free_cpu(p);
    }
};

}

#endif //CPU_ALLOCATOR_H
