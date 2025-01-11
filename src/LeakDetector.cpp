//
// Created by 赵丹 on 25-1-9.
//
#define NEW_OVERLOAD_IMPLEMENTATION

#include "LeakDetector.h"

#include <iostream>
#include <string>

struct Chunk {
    Chunk* next;
    Chunk* prev;
    size_t size;
    bool isArray;
    std::string fileName;
    size_t line;
};

static Chunk head{&head, &head, 0, false, "", 0};

static size_t memoryAllocatedSize = 0;

void* Allocate(size_t size, bool isArray, const char* fileName, size_t line) {
    size_t allocSize = size + sizeof(Chunk);
    auto* p = static_cast<Chunk*>(malloc(allocSize));
    p->prev = &head;
    p->next = head.next;
    p->size = size;
    p->isArray = isArray;
    p->fileName = fileName;
    p->line = line;

    head.next->prev = p;
    head.next = p;
    memoryAllocatedSize += size;

    return (char*)p + sizeof(Chunk);
}

void Deallocate(void* ptr, bool isArray) {
    // auto* p = (Chunk*)(static_cast<char*>(ptr) - sizeof(Chunk));
    auto* p = (Chunk*)((char*)ptr - sizeof(Chunk));
    if (p->isArray != isArray) {
        return;
    }

    p->prev->next = p->next;
    p->next->prev = p->prev;

    memoryAllocatedSize -= p->size;
    free(p);
}

void* operator new(size_t size, const char* file, size_t line) {
    return Allocate(size, false, file, line);
}

// void* operator new(size_t size, std::nothrow_t&, const char* file, size_t line) throw() {
//     return Allocate(size, false, file, line);
// }

void* operator new[](size_t size, const char* file, size_t line) {
    return Allocate(size, true, file, line);
}

// void* operator new[](size_t size, std::nothrow_t&, const char* file, size_t line) throw() {
//     return Allocate(size, true, file, line);
// }

void operator delete(void* ptr, const char* file, size_t line) {
    Deallocate(ptr, false);
}

void operator delete[](void* ptr, const char* file, size_t line) {
    Deallocate(ptr, true);
}


namespace atp {

size_t LeakDetector::instNum = 0;

void LeakDetector::Detect() {
    if (memoryAllocatedSize == 0) {
        std::cout << "There are no leaky memory." << std::endl;
        return;
    }

    size_t cnt = 0;
    auto* p = head.next;
    while (p && p != &head) {
        if (p->isArray) {
            std::cout << "Memory leak detected, allocated by new[] operator, ";
        } else {
            std::cout << "Memory leak detected, allocated by new operator, ";
        }

        std::cout << "pointer: " << p << ", size: " << p->size;

        if (!p->fileName.empty()) {
            std::cout << ", in file: " << p->fileName << ", line: " << p->line;
        } else {
            std::cout << "no file info";
        }

        std::cout << std::endl;

        p = p->next;
        ++cnt;
    }
    std::cout << "There are " << cnt << " memory leaky, total " << memoryAllocatedSize << " Bytes.\n";
}


}
