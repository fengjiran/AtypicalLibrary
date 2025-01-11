//
// Created by 赵丹 on 25-1-9.
//
#define NEW_OVERLOAD_IMPLEMENTATION

#include "LeakDetector.h"

#include <cstring>
#include <iostream>
#include <string>

struct Chunk {
    Chunk* next;
    Chunk* prev;
    size_t size;
    bool isArray;
    char* fileName;
    // std::string fileName;
    size_t line;
};

static Chunk head{&head, &head, 0, false, nullptr, 0};

static size_t memoryAllocatedSize = 0;

void* Allocate(size_t size, bool isArray, const char* fileName, size_t line) {
    size_t allocSize = size + sizeof(Chunk);
    auto* p = (Chunk*) malloc(allocSize);
    // auto* p = static_cast<Chunk*>(malloc(allocSize));
    p->prev = &head;
    p->next = head.next;
    p->size = size;
    p->isArray = isArray;
    // p->fileName = fileName;
    p->line = line;
    if (fileName) {
        p->fileName = (char*) malloc(strlen(fileName) + 1);
        strcpy(p->fileName, fileName);
    } else {
        p->fileName = nullptr;
    }

    head.next->prev = p;
    head.next = p;
    memoryAllocatedSize += size;

    auto ptr = (char*) p + sizeof(Chunk);

    return ptr;
}

void Deallocate(void* ptr, bool isArray) {
    // auto* p = (Chunk*)(static_cast<char*>(ptr) - sizeof(Chunk));
    auto* p = (Chunk*) ((char*) ptr - sizeof(Chunk));
    if (p->isArray != isArray) {
        return;
    }

    p->next->prev = p->prev;
    p->prev->next = p->next;

    memoryAllocatedSize -= p->size;
    if (p->fileName) {
        free(p->fileName);
    }
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

        if (p->fileName) {
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
