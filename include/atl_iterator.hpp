//
// Created by 赵丹 on 24-5-18.
//

#ifndef ATYPICALLIBRARY_ATL_ITERATOR_HPP
#define ATYPICALLIBRARY_ATL_ITERATOR_HPP

#include "utils.hpp"

namespace atp {
template<typename Iter>
class MyIterator {
public:
    using iterator_type = Iter;
    using value_type = typename std::iterator_traits<iterator_type>::value_type;
    using difference_type = typename std::iterator_traits<iterator_type>::difference_type;
    using pointer = typename std::iterator_traits<iterator_type>::pointer;
    using reference = typename std::iterator_traits<iterator_type>::reference;
    using iterator_category = typename std::iterator_traits<iterator_type>::iterator_category;

public:
    MyIterator() noexcept : i() {}

    template<typename U,
             typename std::enable_if<std::is_convertible<U, iterator_type>::value>::type* = nullptr>
    explicit MyIterator(const MyIterator<U>& u) noexcept : i(u.base()) {}

    reference operator*() const noexcept {
        return *i;
    }

    pointer operator->() const noexcept {
        return to_address(i);
        //        static_assert(!std::is_function<value_type>::value, "value is a function type");
        //        return i;
    }

    MyIterator& operator++() noexcept {
        ++i;
        return *this;
    }

    MyIterator operator++(int) noexcept {
        MyIterator tmp(*this);
        ++(*this);
        return tmp;
    }

    MyIterator& operator--() noexcept {
        --i;
        return *this;
    }

    MyIterator operator--(int) noexcept {
        MyIterator tmp(*this);
        --(*this);
        return tmp;
    }

    MyIterator operator+(difference_type n) const noexcept {
        MyIterator tmp(*this);
        tmp += n;
        return tmp;
    }

    MyIterator& operator+=(difference_type n) noexcept {
        i += n;
        return *this;
    }

    MyIterator operator-(difference_type n) const noexcept {
        return *this + (-n);
    }

    MyIterator& operator-=(difference_type n) noexcept {
        *this += (-n);
        return *this;
    }

    reference operator[](difference_type n) const noexcept {
        return i[n];
    }

    iterator_type base() const noexcept {
        return i;
    }

private:
    iterator_type i;

    explicit MyIterator(iterator_type x) noexcept : i(x) {}

    template<typename T, typename Allocator>
    friend class vec;
};

template<typename Iter>
bool operator==(const MyIterator<Iter>& lhs, const MyIterator<Iter>& rhs) noexcept {
    return lhs.base() == rhs.base();
}

template<typename Iter1, typename Iter2>
bool operator==(const MyIterator<Iter1>& lhs, const MyIterator<Iter2>& rhs) noexcept {
    return lhs.base() == rhs.base();
}

template<typename Iter>
bool operator!=(const MyIterator<Iter>& lhs, const MyIterator<Iter>& rhs) noexcept {
    return !(lhs == rhs);
}

template<typename Iter1, typename Iter2>
bool operator!=(const MyIterator<Iter1>& lhs, const MyIterator<Iter2>& rhs) noexcept {
    return !(lhs == rhs);
}

template<typename Iter>
bool operator<(const MyIterator<Iter>& lhs, const MyIterator<Iter>& rhs) noexcept {
    return lhs.base() < rhs.base();
}

template<typename Iter1, typename Iter2>
bool operator<(const MyIterator<Iter1>& lhs, const MyIterator<Iter2>& rhs) noexcept {
    return lhs.base() < rhs.base();
}

template<typename Iter>
bool operator>(const MyIterator<Iter>& lhs, const MyIterator<Iter>& rhs) noexcept {
    return rhs < lhs;
}

template<typename Iter1, typename Iter2>
bool operator>(const MyIterator<Iter1>& lhs, const MyIterator<Iter2>& rhs) noexcept {
    return rhs < lhs;
}

template<typename Iter>
bool operator>=(const MyIterator<Iter>& lhs, const MyIterator<Iter>& rhs) noexcept {
    return !(lhs < rhs);
}

template<typename Iter1, typename Iter2>
bool operator>=(const MyIterator<Iter1>& lhs, const MyIterator<Iter2>& rhs) noexcept {
    return !(lhs < rhs);
}

template<typename Iter>
bool operator<=(const MyIterator<Iter>& lhs, const MyIterator<Iter>& rhs) noexcept {
    return !(rhs < lhs);
}

template<typename Iter1, typename Iter2>
bool operator<=(const MyIterator<Iter1>& lhs, const MyIterator<Iter2>& rhs) noexcept {
    return !(rhs < lhs);
}


template<typename Iter1, typename Iter2>
auto operator-(const MyIterator<Iter1>& lhs, const MyIterator<Iter2>& rhs) noexcept
        -> decltype(lhs.base() - rhs.base()) {
    return lhs.base() - rhs.base();
}

template<typename Iter>
MyIterator<Iter> operator+(typename MyIterator<Iter>::difference_type n, MyIterator<Iter> x) {
    x += n;
    return x;
}

template<typename Iter>
MyIterator<Iter> operator+(MyIterator<Iter> x, typename MyIterator<Iter>::difference_type n) {
    x += n;
    return x;
}
}// namespace atp

#endif//ATYPICALLIBRARY_ATL_ITERATOR_HPP
