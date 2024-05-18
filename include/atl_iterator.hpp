//
// Created by 赵丹 on 24-5-18.
//

#ifndef ATYPICALLIBRARY_ATL_ITERATOR_HPP
#define ATYPICALLIBRARY_ATL_ITERATOR_HPP

#include "utils.hpp"

namespace atp {

template<typename Iter>
class ATLIterator {
public:
    using iterator_type = Iter;
    using value_type = typename std::iterator_traits<iterator_type>::value_type;
    using difference_type = typename std::iterator_traits<iterator_type>::difference_type;
    using pointer = typename std::iterator_traits<iterator_type>::pointer;
    using reference = typename std::iterator_traits<iterator_type>::reference;
    using iterator_category = typename std::iterator_traits<iterator_type>::iterator_category;

public:
    ATLIterator() noexcept : i() {}

    template<typename U,
             typename std::enable_if<std::is_convertible<U, iterator_type>::value>::type* = nullptr>
    explicit ATLIterator(const ATLIterator<U>& u) noexcept : i(u.base()) {}

    reference operator*() const noexcept {
        return *i;
    }

    pointer operator->() const noexcept {
        return to_address(i);
    }

    ATLIterator& operator++() noexcept {
        ++i;
        return *this;
    }

    ATLIterator operator++(int) noexcept {
        ATLIterator tmp(*this);
        ++(*this);
        return tmp;
    }

    ATLIterator& operator--() noexcept {
        --i;
        return *this;
    }

    ATLIterator operator--(int) noexcept {
        ATLIterator tmp(*this);
        --(*this);
        return tmp;
    }

    ATLIterator operator+(difference_type n) const noexcept {
        ATLIterator tmp(*this);
        tmp += n;
        return tmp;
    }

    ATLIterator& operator+=(difference_type n) noexcept {
        i += n;
        return *this;
    }

    ATLIterator operator-(difference_type n) const noexcept {
        return *this + (-n);
    }

    ATLIterator& operator-=(difference_type n) noexcept {
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

    explicit ATLIterator(iterator_type x) noexcept : i(x) {}

    template<typename T, typename Allocator>
    friend class vector;
};

template<typename Iter>
bool operator==(const ATLIterator<Iter>& lhs, const ATLIterator<Iter>& rhs) noexcept {
    return lhs.base() == rhs.base();
}

template<typename Iter1, typename Iter2>
bool operator==(const ATLIterator<Iter1>& lhs, const ATLIterator<Iter2>& rhs) noexcept {
    return lhs.base() == rhs.base();
}

template<typename Iter>
bool operator!=(const ATLIterator<Iter>& lhs, const ATLIterator<Iter>& rhs) noexcept {
    return !(lhs == rhs);
}

template<typename Iter1, typename Iter2>
bool operator!=(const ATLIterator<Iter1>& lhs, const ATLIterator<Iter2>& rhs) noexcept {
    return !(lhs == rhs);
}

template<typename Iter>
bool operator<(const ATLIterator<Iter>& lhs, const ATLIterator<Iter>& rhs) noexcept {
    return lhs.base() < rhs.base();
}

template<typename Iter1, typename Iter2>
bool operator<(const ATLIterator<Iter1>& lhs, const ATLIterator<Iter2>& rhs) noexcept {
    return lhs.base() < rhs.base();
}

template<typename Iter>
bool operator>(const ATLIterator<Iter>& lhs, const ATLIterator<Iter>& rhs) noexcept {
    return rhs < lhs;
}

template<typename Iter1, typename Iter2>
bool operator>(const ATLIterator<Iter1>& lhs, const ATLIterator<Iter2>& rhs) noexcept {
    return rhs < lhs;
}

template<typename Iter>
bool operator>=(const ATLIterator<Iter>& lhs, const ATLIterator<Iter>& rhs) noexcept {
    return !(lhs < rhs);
}

template<typename Iter1, typename Iter2>
bool operator>=(const ATLIterator<Iter1>& lhs, const ATLIterator<Iter2>& rhs) noexcept {
    return !(lhs < rhs);
}

template<typename Iter>
bool operator<=(const ATLIterator<Iter>& lhs, const ATLIterator<Iter>& rhs) noexcept {
    return !(rhs < lhs);
}

template<typename Iter1, typename Iter2>
bool operator<=(const ATLIterator<Iter1>& lhs, const ATLIterator<Iter2>& rhs) noexcept {
    return !(rhs < lhs);
}


template<typename Iter1, typename Iter2>
auto operator-(const ATLIterator<Iter1>& lhs, const ATLIterator<Iter2>& rhs) noexcept
        -> decltype(lhs.base() - rhs.base()) {
    return lhs.base() - rhs.base();
}

template<typename Iter>
ATLIterator<Iter> operator+(typename ATLIterator<Iter>::difference_type n, ATLIterator<Iter> x) {
    x += n;
    return x;
}

template<typename Iter>
ATLIterator<Iter> operator+(ATLIterator<Iter> x, typename ATLIterator<Iter>::difference_type n) {
    x += n;
    return x;
}
}// namespace atp

#endif//ATYPICALLIBRARY_ATL_ITERATOR_HPP
