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
     * @brief Constructs the container with n default-inserted instances of T. No copies are made.
     *
     * @param n Size
     * @param alloc_ The given allocator
     */
    explicit vector(size_type n, const allocator_type& alloc_ = allocator_type());

    /**
     * @brief Constructs the container with n copies of elements with value and the given allocator.
     *
     * @param n Size
     * @param value const reference of initial value
     * @param alloc_ The given allocator
     */
    vector(size_type n, const_reference value, const allocator_type& alloc_ = allocator_type());

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
     * @brief Copy constructor. Constructs the container with the copy of the contents of other.
     *
     * @param other Other vector
     */
    vector(const vector& other);

    /**
     * @brief Constructs the container with the copy of the contents of other, using alloc_ as the allocator.
     *
     * @param other Other vector
     * @param alloc_ The given allocator
     */
    vector(const vector& other, const type_identity_t<allocator_type>& alloc_);

    /**
     * @brief Move constructor. Constructs the container with the contents of other using move semantics.
     *
     * @param other Other vector
     */
    vector(vector&& other) noexcept;

    /**
     * @brief Allocator-extended move constructor. Using alloc_ as the allocator for the new container,
     * moving the contents from other vector.
     *
     * @param other Other vector
     * @param alloc_ The given allocator
     */
    vector(vector&& other, const type_identity_t<allocator_type>& alloc_);

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

    void assign(size_type n, const_reference value);

    template<typename InputIterator,
             typename has_input_iterator_category<InputIterator, value_type>::type = 0>
    void assign(InputIterator first, InputIterator last);

    void assign(std::initializer_list<T> il);

    /**
     * @brief Get the number of elements in the container
     *
     * @return The number of elements in the container.
     */
    size_type size() const { return static_cast<size_type>(firstFree - start); }

    size_type max_size() const noexcept {
        return std::min<size_type>(
                alloc_traits::max_size(_alloc()),
                std::numeric_limits<size_type>::max());
    }

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

    ~vector() noexcept;

private:
    template<typename InputIterator,
             typename has_input_iterator_category<InputIterator, value_type>::type = 0>
    std::pair<pointer, pointer> AllocateWithSentinel(InputIterator first, InputIterator last);

    void _init_with_size(size_type n);

    void _init_with_size(size_type n, const_reference value);

    template<typename InputIterator,
             typename has_input_iterator_category<InputIterator, value_type>::type = 0>
    void _init_with_range(InputIterator first, InputIterator last);

    /**
     * @brief Allocate space for n objects.
     *
     * @param n The number of objects to be allocated.
     *
     * @Precondition: start == firstFree == cap == nullptr
     * @Precondition: n > 0
     *
     * @Postcondition: capacity() >= n
     * @Postcondition: size() == 0
     */
    void _allocate_with_size(size_type n);

    void _construct_at_end(size_type n);

    void _construct_at_end(size_type n, const_reference value);

    template<typename InputIterator,
             typename has_input_iterator_category<InputIterator, value_type>::type = 0>
    void _construct_at_end(InputIterator first, InputIterator last);

    template<typename... Args>
    void _construct_one_at_end(Args&&... args) {
        alloc_traits::construct(alloc, firstFree++, std::forward<Args>(args)...);
    }

    void _destruct_at_end(pointer new_last) noexcept {
        pointer p = firstFree;
        while (p != new_last) {
            alloc_traits::destroy(_alloc(), to_address(--p));
        }
        firstFree = new_last;
    }

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
        _destruct_at_end(start);
    }

    void _copy_assign_alloc(const vector& c, true_type) {
        if (_alloc() != c._alloc()) {
            _destroy_vector (*this)();
            start = nullptr;
            firstFree = nullptr;
            cap = nullptr;
        }
        _alloc() = c._alloc();
    }

    void _copy_assign_alloc(const vector&, false_type) {}

    void _move_assign(vector& rhs, true_type) {
        _destroy_vector (*this)();
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

    size_type _recommend_size(size_type new_size) const {
        const size_type ms = max_size();
        if (new_size > ms) {
            throw std::bad_array_new_length();
        }

        if (ms / 2 <= capacity()) {
            return ms;
        }
        return std::max<size_type>(2 * capacity(), new_size);
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
vector<T, Allocator>::vector(size_type n, const allocator_type& alloc_)
    : alloc(alloc_) {
    _init_with_size(n);
}

template<typename T, typename Allocator>
vector<T, Allocator>::vector(size_type n, const_reference value, const allocator_type& alloc_)
    : alloc(alloc_) {
    _init_with_size(n, value);
}

template<typename T, typename Allocator>
template<typename InputIterator,
         typename has_input_iterator_category<InputIterator, T>::type>
vector<T, Allocator>::vector(InputIterator first, InputIterator last, const allocator_type& alloc_)
    : alloc(alloc_) {
    _init_with_range(first, last);
}

template<typename T, typename Allocator>
vector<T, Allocator>::vector(const vector<T, Allocator>& other)
    : alloc(select_on_container_copy_construction(other._alloc())) {
    _init_with_range(other.begin(), other.end());
}

template<typename T, typename Allocator>
vector<T, Allocator>::vector(const vector<T, Allocator>& other, const type_identity_t<allocator_type>& alloc_)
    : alloc(alloc_) {
    _init_with_range(other.begin(), other.end());
}

template<typename T, typename Allocator>
vector<T, Allocator>::vector(vector&& other) noexcept
    : start(other.start), firstFree(other.firstFree), cap(other.cap), alloc(std::move(other.alloc)) {
    other.start = nullptr;
    other.firstFree = nullptr;
    other.cap = nullptr;
}

template<typename T, typename Allocator>
vector<T, Allocator>::vector(vector&& other, const type_identity_t<allocator_type>& alloc_)
    : alloc(alloc_) {
    if (other._alloc() == alloc_) {
        start = other.start;
        firstFree = other.firstFree;
        cap = other.cap;
    } else {
        auto guard = _make_exception_guard(_destroy_vector(*this));
        for (auto iter = other.begin(); iter != other.end(); ++iter) {
            emplace_back(std::move(*iter));
        }
        guard.complete();
    }

    other.start = nullptr;
    other.firstFree = nullptr;
    other.cap = nullptr;
}

template<typename T, typename Allocator>
vector<T, Allocator>::vector(std::initializer_list<T> il, const allocator_type& alloc_)
    : alloc(alloc_) {
    _init_with_range(il.begin(), il.end());
}

template<typename T, typename Allocator>
vector<T, Allocator>& vector<T, Allocator>::operator=(std::initializer_list<T> il) {
    assign(il.begin(), il.end());
    return *this;
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
vector<T, Allocator>& vector<T, Allocator>::operator=(vector&& rhs) noexcept {
    _move_assign(rhs,
                 integral_constant<bool,
                                   propagate_on_container_move_assignment<Allocator>::type::value>());
    return *this;
}

template<typename T, typename Allocator>
void vector<T, Allocator>::assign(size_type n, const_reference value) {
    if (n <= capacity()) {
        size_type s = size();
        std::fill_n(start, std::min(s, n), value);
        if (n > s) {
            _construct_at_end(n - s, value);
        } else {
            _destruct_at_end(start + n);
        }
    } else {
        _destroy_vector (*this)();
        _allocate_with_size(_recommend_size(n));
        _construct_at_end(n, value);
    }
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
void vector<T, Allocator>::assign(std::initializer_list<T> il) {
    assign(il.begin(), il.end());
}

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
vector<T, Allocator>::~vector() noexcept {
    _destroy_vector (*this)();
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
    _destroy_vector (*this)();
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
    _destroy_vector (*this)();
    start = data;
    firstFree = dst;
    cap = start + newCap;
}


template<typename T, typename Allocator>
template<typename InputIterator,
         typename has_input_iterator_category<InputIterator, T>::type>
std::pair<typename vector<T, Allocator>::pointer, typename vector<T, Allocator>::pointer>
vector<T, Allocator>::AllocateWithSentinel(InputIterator first, InputIterator last) {
    auto guard = _make_exception_guard(_destroy_vector(*this));
    auto n = std::distance(first, last);
    auto begin = alloc_traits::allocate(alloc, n);
    auto end = std::uninitialized_copy(first, last, begin);
    guard.complete();

    return {begin, end};
}

template<typename T, typename Allocator>
void vector<T, Allocator>::_init_with_size(size_type n) {
    auto guard = _make_exception_guard(_destroy_vector(*this));
    if (n > 0) {
        _allocate_with_size(n);
        _construct_at_end(n);
    }
    guard.complete();
}

template<typename T, typename Allocator>
void vector<T, Allocator>::_init_with_size(size_type n, const_reference value) {
    auto guard = _make_exception_guard(_destroy_vector(*this));
    if (n > 0) {
        _allocate_with_size(n);
        _construct_at_end(n, value);
    }
    guard.complete();
}

template<typename T, typename Allocator>
template<typename InputIterator,
         typename has_input_iterator_category<InputIterator, T>::type>
void vector<T, Allocator>::_init_with_range(InputIterator first, InputIterator last) {
    auto guard = _make_exception_guard(_destroy_vector(*this));
    auto n = std::distance(first, last);
    _allocate_with_size(n);
    _construct_at_end(first, last);
    guard.complete();
}

template<typename T, typename Allocator>
void vector<T, Allocator>::_allocate_with_size(size_type n) {
    start = alloc_traits::allocate(alloc, n);
    firstFree = start;
    cap = start + n;
}

template<typename T, typename Allocator>
void vector<T, Allocator>::_construct_at_end(size_type n) {
    for (size_type i = 0; i < n; ++i) {
        alloc_traits::construct(alloc, firstFree++);
    }
}

template<typename T, typename Allocator>
void vector<T, Allocator>::_construct_at_end(size_type n, const_reference value) {
    for (size_type i = 0; i < n; ++i) {
        alloc_traits::construct(alloc, firstFree++, value);
    }
}

template<typename T, typename Allocator>
template<typename InputIterator,
         typename has_input_iterator_category<InputIterator, T>::type>
void vector<T, Allocator>::_construct_at_end(InputIterator first, InputIterator last) {
    firstFree = std::uninitialized_copy(first, last, firstFree);
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
