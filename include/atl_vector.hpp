//
// Created by 赵丹 on 24-5-18.
//

#ifndef ATYPICALLIBRARY_ATL_VECTOR_HPP
#define ATYPICALLIBRARY_ATL_VECTOR_HPP

#include "atl_allocator.hpp"
#include "atl_iterator.hpp"
#include "config.hpp"
#include "exception_guard.hpp"
#include "glog/logging.h"
#include "utils.hpp"

namespace atp {

template<typename T, typename Allocator = ATLAllocator<T> /* = std::allocator<T>*/>
class vector {
public:
    using value_type = T;
    using allocator_type = Allocator;
    using alloc_traits = std::allocator_traits<allocator_type>;
    using size_type = typename alloc_traits::size_type;
    using difference_type = typename alloc_traits::difference_type;
    using pointer = typename alloc_traits::pointer;
    using const_pointer = typename alloc_traits::const_pointer;
    using reference = value_type&;
    using const_reference = const value_type&;

    using iterator = ATLIterator<pointer>;
    using const_iterator = ATLIterator<const_pointer>;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    static_assert((std::is_same<typename allocator_type::value_type, value_type>::value),
                  "Allocator::value_type must be same type as value_type");

public:
    /**
     * @brief Default constructor.
     *
     * Constructs an empty container with a default-constructed allocator.
     */
    vector() noexcept(noexcept(allocator_type()))
        : start(nullptr), cap(nullptr), firstFree(nullptr), alloc(allocator_type()) {}

    /**
     * @brief Constructs an empty container with the given allocator.
     *
     * @param alloc_ The given allocator.
     */
    explicit vector(const allocator_type& alloc_) noexcept
        : start(nullptr), cap(nullptr), firstFree(nullptr), alloc(alloc_) {}

    /**
     * @brief Constructs the container with n copies of elements with value.
     *
     * @param n Size
     * @param value const reference of initial value
     * @param alloc_ The given allocator
     */
    vector(size_type n, const_reference value, const allocator_type& alloc_ = allocator_type());

    /**
     * @brief Constructs the container with n default-inserted instances of T. No copies are made.
     *
     * @param n Size
     * @param alloc_ The given allocator
     */
    explicit vector(size_type n, const allocator_type& alloc_ = allocator_type());

    /**
     * @brief Constructs the container with the contents of the range [first, last).
     *
     * @param first First iterator
     * @param last Last iterator
     * @param alloc_ The given allocator
     */
    template<typename InputIterator,
             typename has_input_iterator_category<InputIterator, value_type>::type = 0>
    vector(InputIterator first, InputIterator last, const allocator_type& alloc_ = allocator_type());

    /**
     * @brief Copy constructor. Constructs the container with the copy of the contents of rhs.
     *
     * @param rhs Right Hand Side
     */
    vector(const vector& rhs);

    /**
     * @brief Constructs the container with the copy of the contents of rhs, using alloc_ as the allocator.
     *
     * @param rhs Right Hand Side
     * @param alloc_ The given allocator
     */
    vector(const vector& rhs, const type_identity_t<allocator_type>& alloc_);

    /**
     * @brief Move constructor. Constructs the container with the contents of rhs using move semantics.
     *
     * @param rhs Right Hand Side
     */
    vector(vector&& rhs) noexcept;

    /**
     * @brief Allocator-extended move constructor. Using alloc_ as the allocator for the new container,
     * moving the contents from rhs.
     *
     * @param rhs Right Hand Side
     * @param alloc_ The given allocator
     */
    vector(vector&& rhs, const type_identity_t<allocator_type>& alloc_);

    /**
     * @brief Constructs the container with the contents of the initializer list.
     *
     * @param il Initializer list
     * @param alloc_ The given allocator
     */
    vector(std::initializer_list<T> il, const allocator_type& alloc_ = allocator_type());

    vector& operator=(std::initializer_list<T> il);

    /**
     * @brief Copy assignment operator
     *
     * @param rhs Right Hand Side
     * @return The self reference
     */
    vector& operator=(const vector& rhs);

    /**
     * @brief Move assignment
     *
     * @param rhs Right Hand Side
     * @return The self reference
     */
    vector& operator=(vector&& rhs) noexcept;

    /**
     * @brief Get the number of elements in the container
     *
     * @return The number of elements in the container.
     */
    size_type size() const { return static_cast<size_type>(firstFree - start); }

    /**
     * @brief Get the number of elements that the container has currently allocated space for
     *
     * @return Capacity of the currently allocated storage.
     */
    size_type capacity() const { return static_cast<size_type>(cap - start); }

    /**
     * @brief Get the first element iterator of the vector
     *
     * @return The pointer of the first element
     */
    iterator begin() noexcept { return _make_iter(start); }

    const_iterator begin() const noexcept { return _make_iter(start); }

    /**
     * @brief Get the ptr to the element following the last element of the vector
     *
     * @return The ptr to the element following the last element of the vector
     */
    iterator end() noexcept { return _make_iter(firstFree); }

    const_iterator end() const noexcept { return _make_iter(firstFree); }

    CPP_NODISCARD bool empty() const { return firstFree == start; }

    void reserve(size_type n);
    void resize(size_type n);
    void resize(size_type n, const_reference t);
    pointer data() noexcept { return to_address(start); }
    const_pointer data() const noexcept { return to_address(start); }

    void push_back(const_reference t);
    void push_back(value_type&& t);

    template<typename... Args>
    void emplace_back(Args&&... args);

    reference operator[](size_type pos) noexcept;
    const_reference operator[](size_type pos) const noexcept;

    reference at(size_type pos);
    const_reference at(size_type pos) const;

    reference front() noexcept;
    const_reference front() const noexcept;

    /**
     * @brief Get the reference to the last element in the container.
     *
     * @return Reference to the last element.
     */
    reference back() noexcept;

    /**
     * @brief Get the reference to the last element in the container.
     *
     * @return Reference to the last element.
     */
    const_reference back() const noexcept;

    allocator_type get_allocator() const noexcept {
        return _alloc();
    }

    void clear() noexcept {
        _clear();
    }

    template<typename InputIterator,
             typename has_input_iterator_category<InputIterator, value_type>::type = 0>
    void assign(InputIterator first, InputIterator last);

    ~vector();

private:
    template<typename InputIterator,
             typename has_input_iterator_category<InputIterator, value_type>::type = 0>
    std::pair<pointer, pointer> Allocate(InputIterator first, InputIterator last);

    void _free();
    void _reallocate();
    void _reallocate(size_type newCap);
    void _check_and_alloc() {
        if (firstFree == cap) {
            _reallocate();
        }
    }

    iterator _make_iter(pointer p) noexcept {
        return iterator(p);
    }

    const_iterator _make_iter(const_pointer p) const noexcept {
        return const_iterator(p);
    }

    allocator_type& _alloc() noexcept {
        return alloc;
    }

    const allocator_type& _alloc() const noexcept {
        return alloc;
    }

    void _clear() noexcept {
        auto p = firstFree;
        while (p != start) {
            alloc_traits::destroy(_alloc(), to_address(--p));
        }
        firstFree = start;
    }

    void _copy_assign_alloc(const vector& c, true_type) {
        if (_alloc() != c._alloc()) {
            _free();
            start = nullptr;
            firstFree = nullptr;
            cap = nullptr;
        }
        _alloc() = c._alloc();
    }

    void _copy_assign_alloc(const vector&, false_type) {}

    void _move_assign(vector& rhs, true_type) {
        _free();
        _alloc() = std::move(rhs._alloc());

        start = rhs.start;
        firstFree = rhs.firstFree;
        cap = rhs.cap;

        rhs.start = nullptr;
        rhs.firstFree = nullptr;
        rhs.cap = nullptr;
    }

    void _move_assign(vector& rhs, false_type) {
        if (_alloc() != rhs._alloc()) {
            _clear();
            for (auto iter = rhs.begin(); iter != rhs.end(); ++iter) {
                emplace_back(std::move(*iter));
            }
            rhs.start = nullptr;
            rhs.firstFree = nullptr;
            rhs.cap = nullptr;
        } else {
            _move_assign(rhs, true_type());
        }
    }

    class _destroy_vector {
    public:
        explicit _destroy_vector(vector& _vec) : _vec_(_vec) {}

        void operator()() {
            if (_vec_.start) {
                _vec_._clear();
                alloc_traits::deallocate(_vec_._alloc(), _vec_.start, _vec_.capacity());
            }
        }

    private:
        vector& _vec_;
    };

private:
    allocator_type alloc;
    pointer start;
    pointer cap;
    pointer firstFree;
};

template<typename T, typename Allocator>
typename vector<T, Allocator>::const_reference
vector<T, Allocator>::back() const noexcept {
    CHECK(!empty()) << "back() called on an empty vector";
    return *(firstFree - 1);
}

template<typename T, typename Allocator>
typename vector<T, Allocator>::reference
vector<T, Allocator>::back() noexcept {
    CHECK(!empty()) << "back() called on an empty vector";
    return *(firstFree - 1);
}

template<typename T, typename Allocator>
typename vector<T, Allocator>::const_reference
vector<T, Allocator>::front() const noexcept {
    CHECK(!empty()) << "front() called on an empty vector";
    return *start;
}

template<typename T, typename Allocator>
typename vector<T, Allocator>::reference
vector<T, Allocator>::front() noexcept {
    CHECK(!empty()) << "front() called on an empty vector";
    return *start;
}

template<typename T, typename Allocator>
typename vector<T, Allocator>::const_reference
vector<T, Allocator>::operator[](size_type pos) const noexcept {
    CHECK(pos < size()) << "vector[] index out of bounds";
    return start[pos];
}

template<typename T, typename Allocator>
typename vector<T, Allocator>::reference
vector<T, Allocator>::operator[](size_type pos) noexcept {
    CHECK(pos < size()) << "vector[] index out of bounds";
    return start[pos];
}

template<typename T, typename Allocator>
typename vector<T, Allocator>::reference
vector<T, Allocator>::at(size_type pos) {
    if (pos >= size()) {
        throw std::out_of_range("index out of bounds");
    }
    return (*this)[pos];
}

template<typename T, typename Allocator>
typename vector<T, Allocator>::const_reference
vector<T, Allocator>::at(size_type pos) const {
    if (pos >= size()) {
        throw std::out_of_range("index out of bounds");
    }
    return (*this)[pos];
}

template<typename T, typename Allocator>
template<typename... Args>
void vector<T, Allocator>::emplace_back(Args&&... args) {
    _check_and_alloc();
    alloc_traits::construct(alloc, firstFree++, std::forward<Args>(args)...);
}

template<typename T, typename Allocator>
void vector<T, Allocator>::push_back(value_type&& t) {
    _check_and_alloc();
    alloc_traits::construct(alloc, firstFree++, std::move(t));
}

template<typename T, typename Allocator>
void vector<T, Allocator>::push_back(const_reference t) {
    _check_and_alloc();
    alloc_traits::construct(alloc, firstFree++, t);
}

template<typename T, typename Allocator>
void vector<T, Allocator>::resize(size_type n) {
    if (n > size()) {
        while (size() < n) {
            push_back(T());
        }
    } else {
        while (size() > n) {
            alloc_traits::destroy(alloc, --firstFree);
        }
    }
}

template<typename T, typename Allocator>
void vector<T, Allocator>::resize(size_type n, const_reference t) {
    if (n > size()) {
        while (size() < n) {
            push_back(t);
        }
    }
}

template<typename T, typename Allocator>
void vector<T, Allocator>::reserve(size_type n) {
    if (n > capacity()) {
        _reallocate(n);
    }
}

template<typename T, typename Allocator>
vector<T, Allocator>::~vector() {
    _free();
}

template<typename T, typename Allocator>
void vector<T, Allocator>::_reallocate(size_type newCap) {
    auto data = alloc_traits::allocate(alloc, newCap);
    auto src = start;
    auto dst = data;
    for (size_type i = 0; i < size(); ++i) {
        alloc_traits::construct(alloc, dst, std::move(*src));
        ++src;
        ++dst;
    }
    _free();
    start = data;
    firstFree = dst;
    cap = start + newCap;
}

template<typename T, typename Allocator>
void vector<T, Allocator>::_reallocate() {
    size_type newCap = size() != 0 ? 2 * size() : 1;
    auto data = alloc_traits::allocate(alloc, newCap);
    auto src = start;
    auto dst = data;
    for (size_type i = 0; i < size(); ++i) {
        alloc_traits::construct(alloc, dst, std::move(*src));
        ++src;
        ++dst;
    }
    _free();
    start = data;
    firstFree = dst;
    cap = start + newCap;
}

template<typename T, typename Allocator>
vector<T, Allocator>& vector<T, Allocator>::operator=(vector&& rhs) noexcept {
    _move_assign(rhs,
                 integral_constant<bool,
                                   propagate_on_container_move_assignment<Allocator>::type::value>());
    return *this;
}

template<typename T, typename Allocator>
vector<T, Allocator>::vector(vector&& rhs) noexcept
    : start(rhs.start), firstFree(rhs.firstFree), cap(rhs.cap), alloc(std::move(rhs.alloc)) {
    rhs.start = nullptr;
    rhs.firstFree = nullptr;
    rhs.cap = nullptr;
}

template<typename T, typename Allocator>
vector<T, Allocator>::vector(vector&& rhs, const type_identity_t<allocator_type>& alloc_)
    : alloc(alloc_) {
    if (rhs._alloc() == alloc_) {
        start = rhs.start;
        firstFree = rhs.firstFree;
        cap = rhs.cap;
    } else {
        for (auto iter = rhs.begin(); iter != rhs.end(); ++iter) {
            emplace_back(std::move(*iter));
        }
    }

    rhs.start = nullptr;
    rhs.firstFree = nullptr;
    rhs.cap = nullptr;
}

template<typename T, typename Allocator>
vector<T, Allocator>::vector(size_type n, const allocator_type& alloc_)
    : alloc(alloc_) {
    // todo: add exception guard
    auto data = alloc_traits::allocate(alloc, n);
    start = data;
    firstFree = data;
    cap = start + n;
    for (size_type i = 0; i < n; ++i) {
        alloc_traits::construct(alloc, firstFree++, T());
    }
}

template<typename T, typename Allocator>
vector<T, Allocator>::vector(size_type n, const_reference value, const allocator_type& alloc_)
    : alloc(alloc_) {
    // todo: add exception guard
    auto data = alloc_traits::allocate(alloc, n);
    start = data;
    firstFree = data;
    cap = start + n;
    for (size_type i = 0; i < n; ++i) {
        alloc_traits::construct(alloc, firstFree++, value);
    }
}

template<typename T, typename Allocator>
template<typename InputIterator,
         typename has_input_iterator_category<InputIterator, T>::type>
vector<T, Allocator>::vector(InputIterator first, InputIterator last, const allocator_type& alloc_)
    : alloc(alloc_) {
    auto data = Allocate(first, last);
    start = data.first;
    firstFree = data.second;
    cap = data.second;
}

template<typename T, typename Allocator>
vector<T, Allocator>::vector(std::initializer_list<T> il, const allocator_type& alloc_)
    : alloc(alloc_) {
    auto data = Allocate(il.begin(), il.end());
    start = data.first;
    firstFree = data.second;
    cap = data.second;
}

template<typename T, typename Allocator>
vector<T, Allocator>& vector<T, Allocator>::operator=(std::initializer_list<T> il) {
    auto data = Allocate(il.begin(), il.end());
    _free();
    start = data.first;
    firstFree = data.second;
    cap = data.second;
    return *this;
}

template<typename T, typename Allocator>
vector<T, Allocator>::vector(const vector<T, Allocator>& rhs)
    : alloc(select_on_container_copy_construction(rhs._alloc())) {
    auto data = Allocate(rhs.begin(), rhs.end());
    start = data.first;
    firstFree = data.second;
    cap = data.second;
}

template<typename T, typename Allocator>
vector<T, Allocator>::vector(const vector<T, Allocator>& rhs, const type_identity_t<allocator_type>& alloc_)
    : alloc(alloc_) {
    auto data = Allocate(rhs.begin(), rhs.end());
    start = data.first;
    firstFree = data.second;
    cap = data.second;
}

template<typename T, typename Allocator>
vector<T, Allocator>& vector<T, Allocator>::operator=(const vector<T, Allocator>& rhs) {
    if (this != std::addressof(rhs)) {
        _copy_assign_alloc(
                rhs,
                integral_constant<bool,
                                  propagate_on_container_copy_assignment<Allocator>::type::value>());
        assign(rhs.begin(), rhs.end());
    }
    return *this;
}

template<typename T, typename Allocator>
template<typename InputIterator,
         typename has_input_iterator_category<InputIterator, T>::type>
std::pair<typename vector<T, Allocator>::pointer, typename vector<T, Allocator>::pointer>
vector<T, Allocator>::Allocate(InputIterator first, InputIterator last) {
    auto n = std::distance(first, last);
    auto dst = alloc_traits::allocate(alloc, n);
    return {dst, std::uninitialized_copy(first, last, dst)};
}

template<typename T, typename Allocator>
template<typename InputIterator,
         typename has_input_iterator_category<InputIterator, T>::type>
void vector<T, Allocator>::assign(InputIterator first, InputIterator last) {
    _clear();
    while (first != last) {
        emplace_back(*first);
        ++first;
    }
}

template<typename T, typename Allocator>
void vector<T, Allocator>::_free() {
    if (start) {
        auto p = firstFree;
        while (p != start) {
            alloc_traits::destroy(_alloc(), to_address(--p));
        }
        alloc_traits::deallocate(_alloc(), start, capacity());
    }
}

template<typename T, typename Allocator>
bool operator==(const vector<T, Allocator>& x, const vector<T, Allocator>& y) {
    return x.size() == y.size() && std::equal(x.begin(), x.end(), y.begin());
}

template<typename T, typename Allocator>
bool operator!=(const vector<T, Allocator>& x, const vector<T, Allocator>& y) {
    return !(x == y);
}

template<typename T, typename Allocator>
std::ostream& operator<<(std::ostream& s, const vector<T, Allocator>& v) {
    s.put('{');

    //    // Range-based for loop initialization statements(c++ 20)
    //    for (char comma[]{'\0', ' ', '\0'}; const auto& e : v) {
    //        s << comma << e;
    //        comma[0] = ',';
    //    }

    char comma[]{'\0', ' ', '\0'};
    for (const auto& e: v) {
        s << comma << e;
        comma[0] = ',';
    }
    return s << "}\n";
}
}// namespace atp


#endif//ATYPICALLIBRARY_ATL_VECTOR_HPP
