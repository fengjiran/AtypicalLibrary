//
// Created by richard on 11/6/24.
//

#ifndef ATL_CONSTRUCT_H
#define ATL_CONSTRUCT_H

namespace atp {

/**
 * Constructs an object in existing memory by invoking an allocated
 * object's constructor with an initializer.
 */
template<typename T, typename... Args>
void construct_at(T* p, Args&&... args) {
    ::new (static_cast<void*>(p)) T(std::forward<Args>(args)...);
}

template<typename T>
void construct_novalue_at(T* p) {
    ::new (static_cast<void*>(p)) T;
}

template<typename T>
void destroy_at(T* p) {
    p->~T();
}

template<bool>
struct Destroy_aux {
    template<typename ForwardIterator>
    static void _destroy(ForwardIterator first, ForwardIterator last) {
        while (first != last) {
            destroy_at(std::addressof(*first));
            ++first;
        }
    }
};

template<>
struct Destroy_aux<true> {
    template<typename ForwardIterator>
    static void _destroy(ForwardIterator, ForwardIterator) {}
};

/**
 * Destroy a range of objects.  If the value_type of the object has
 * a trivial destructor, the compiler should optimize all of this
 * away, otherwise the objects' destructors must be invoked.
 */
template<typename ForwardIterator>
void Destroy(ForwardIterator first, ForwardIterator last) {
    using value_type = typename std::iterator_traits<ForwardIterator>::value_type;
    static_assert(std::is_destructible_v<value_type>, "value type is destructible");
    Destroy_aux<__has_trivial_destructor(value_type)>::_destroy(first, last);
}

template<bool>
struct Destroy_n_aux {
    template<typename ForwardIterator, typename Size>
    static ForwardIterator _destroy_n(ForwardIterator first, Size n) {
        while (n > 0) {
            destroy_at(std::addressof(*first));
            ++first;
            --n;
        }
        return first;
    }
};

template<>
struct Destroy_n_aux<true> {
    template<typename ForwardIterator, typename Size>
    static ForwardIterator _destroy_n(ForwardIterator first, Size n) {
        std::advance(first, n);
        return first;
    }
};

/**
 * Destroy a range of objects.  If the value_type of the object has
 * a trivial destructor, the compiler should optimize all of this
 * away, otherwise the objects' destructors must be invoked.
 */
template<typename ForwardIterator, typename Size>
ForwardIterator Destroy_n(ForwardIterator first, Size n) {
    using value_type = typename std::iterator_traits<ForwardIterator>::value_type;
    static_assert(std::is_destructible_v<value_type>, "value type is destructible");
    return Destroy_n_aux<__has_trivial_destructor(value_type)>::_destroy_n(first, n);
}

template<typename ForwardIterator>
void destroy(ForwardIterator first, ForwardIterator last) {
    Destroy(first, last);
}

template<typename ForwardIterator, typename Size>
ForwardIterator destroy_n(ForwardIterator first, Size n) {
    return Destroy_n(first, n);
}

}// namespace atp

#endif//ATL_CONSTRUCT_H
