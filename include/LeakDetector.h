//
// Created by 赵丹 on 25-1-9.
//

#ifndef LEAKDETECTOR_H
#define LEAKDETECTOR_H

#include <cstdio>
#include <new>
// #include <string>
//
// struct Chunk {
//     Chunk* next;
//     Chunk* prev;
//     size_t size;
//     bool isArray;
//     std::string fileName;
//     size_t line;
// };

void* operator new(size_t size, const char* file, size_t line);
void* operator new[](size_t size, const char* file, size_t line);

void operator delete(void* ptr);
void operator delete(void* ptr, const char* file, size_t line);
void operator delete[](void* ptr);
void operator delete[](void* ptr, const char* file, size_t line);

#ifndef NEW_OVERLOAD_IMPLEMENTATION
#define new new (__FILE__, __LINE__)
#endif

namespace atp {

class LeakDetector {
public:
    static size_t instNum;
    LeakDetector() {
        ++instNum;
    }

    ~LeakDetector() {
        --instNum;
        if (instNum == 0) {
            Detect();
        }
    }

private:
    static void Detect();
};

static LeakDetector leakDetector;

}// namespace atp

#endif//LEAKDETECTOR_H
