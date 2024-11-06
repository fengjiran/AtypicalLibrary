//
// Created by richard on 11/6/24.
//

#ifndef ATL_TEMPBUF_H
#define ATL_TEMPBUF_H

#include "atl_construct.h"

namespace atp {

template<typename T>
std::pair<T*, std::ptrdiff_t> get_temporary_buffer(std::ptrdiff_t len) noexcept {
    const std::ptrdiff_t maxLen = std::numeric_limits<std::ptrdiff_t>::max() / sizeof(T);
    if (len > maxLen) {
        len = maxLen;
    }

    while (len > 0) {
        T* tmp = static_cast<T*>(::operator new(len * sizeof(T), std::nothrow));
        if (tmp) {
            return std::pair<T*, std::ptrdiff_t>(tmp, len);
        }
        len = len == 1 ? 0 : (len + 1) / 2;
    }

    return std::pair<T*, std::ptrdiff_t>(static_cast<T*>(nullptr), 0);
}

template<typename T>
void return_temporary_buffer(T* p) {
    ::operator delete(p);
}

template<typename ForwardIter, typename T>
class TemporaryBuffer {
public:
    using value_type = T;
    using pointer = value_type*;
    using iterator = pointer;
    using size_type = std::ptrdiff_t;

    TemporaryBuffer(ForwardIter seed, size_type original_len);

protected:
    size_type m_original_len;
    size_type m_len;
    pointer m_buffer;
};

template<typename ForwardIter, typename T>
TemporaryBuffer<ForwardIter, T>::TemporaryBuffer(ForwardIter seed, size_type original_len)
    : m_original_len(original_len), m_len(0), m_buffer(nullptr) {
    std::pair<pointer, size_type> p(get_temporary_buffer<T>(original_len));
    if (p.first) {
        //
    }
}


}// namespace atp


#endif//ATL_TEMPBUF_H
