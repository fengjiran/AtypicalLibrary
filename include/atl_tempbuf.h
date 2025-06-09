//
// Created by richard on 11/6/24.
//

#ifndef ATL_TEMPBUF_H
#define ATL_TEMPBUF_H

#include "atl_construct.h"

#include <algorithm>
#include <limits>
// #include <numeric>
#include <map>

namespace atp {

template<typename T>
std::pair<T*, std::ptrdiff_t> get_temporary_buffer(std::ptrdiff_t len) noexcept {
    const std::ptrdiff_t maxLen = std::numeric_limits<std::ptrdiff_t>::max() / sizeof(T);
    if (len > maxLen) {
        len = maxLen;
    }

    while (len > 0) {
        if (T* tmp = static_cast<T*>(::operator new(len * sizeof(T), std::nothrow))) {
            return std::make_pair(tmp, len);
            // return std::pair<T*, std::ptrdiff_t>(tmp, len);
        }
        len = len == 1 ? 0 : (len + 1) / 2;
    }

    // return std::pair<T*, std::ptrdiff_t>(static_cast<T*>(nullptr), 0);
    return std::make_pair(static_cast<T*>(nullptr), 0);
}

template<typename T>
void return_temporary_buffer(T* p) {
    ::operator delete(p);
}

template<typename T>
void return_temporary_buffer(T* p, size_t len) {
    ::operator delete(p, len * sizeof(T));
}

template<typename ForwardIter, typename T>
class TemporaryBuffer {
public:
    using value_type = T;
    using pointer = value_type*;
    using iterator = pointer;
    using size_type = std::ptrdiff_t;

    TemporaryBuffer(ForwardIter seed, size_type original_len);

    TemporaryBuffer(const TemporaryBuffer&) = delete;

    TemporaryBuffer& operator=(const TemporaryBuffer&) = delete;

    size_type size() const {
        return m_len;
    }

    size_type requested_size() const {
        return m_original_len;
    }

    iterator begin() {
        return m_buffer;
    }

    iterator end() {
        return m_buffer + m_len;
    }

    ~TemporaryBuffer() {
        destroy(m_buffer, m_buffer + m_len);
        return_temporary_buffer(m_buffer, m_len);
    }

protected:
    size_type m_original_len;
    size_type m_len;
    pointer m_buffer;
};

template<bool>
struct uninitialized_construct_buf_dispatch {
    template<typename Pointer, typename ForwardIterator>
    static void ucr(Pointer first, Pointer last, ForwardIterator seed) {
        if (first == last) {
            return;
        }

        Pointer cur = first;
        try {
            construct_at(std::addressof(*first), std::move(*seed));
            Pointer prev = cur;
            ++cur;

            while (cur != last) {
                construct_at(std::addressof(*cur), std::move(*prev));
                ++cur;
                ++prev;
            }
            *seed = std::move(*prev);
        } catch (...) {
            destroy(first, cur);
            throw;
        }
    }
};

template<>
struct uninitialized_construct_buf_dispatch<true> {
    template<typename Pointer, typename ForwardIterator>
    static void ucr(Pointer first, Pointer last, ForwardIterator seed) {}
};

// Constructs objects in the range [first, last).
// Note that while these new objects will take valid values,
// their exact value is not defined. In particular they may
// be 'moved from'.
//
// While *__seed may be altered during this algorithm, it will have
// the same value when the algorithm finishes, unless one of the
// constructions throws.
//
// Requirements: _Pointer::value_type(_Tp&&) is valid.
template<typename Pointer, typename ForwardIterator>
void uninitialized_construct_buf(Pointer first, Pointer last, ForwardIterator seed) {
    using value_type = typename std::iterator_traits<Pointer>::value_type;
    uninitialized_construct_buf_dispatch<__has_trivial_constructor(value_type)>::ucr(
            first, last, seed);
}

template<typename ForwardIter, typename T>
TemporaryBuffer<ForwardIter, T>::TemporaryBuffer(ForwardIter seed, size_type original_len)
    : m_original_len(original_len), m_len(0), m_buffer(nullptr) {
    std::pair<pointer, size_type> p = get_temporary_buffer<T>(original_len);

    if (p.first) {
        try {
            uninitialized_construct_buf(p.first, p.first + p.second, seed);
            m_buffer = p.first;
            m_len = p.second;
        } catch (...) {
            return_temporary_buffer(p.first, p.second);
            throw;
        }
    }
}


}// namespace atp


#endif//ATL_TEMPBUF_H
