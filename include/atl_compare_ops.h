//
// Created by richard on 11/13/24.
//

#ifndef ATL_COMPARE_OPS_H
#define ATL_COMPARE_OPS_H

namespace atp {

struct Iter_less_iter {
    template<typename iterator1, typename iterator2>
    constexpr bool operator()(iterator1 it1, iterator2 it2) const {
        return *it1 < *it2;
    }
};

constexpr Iter_less_iter iter_less_iter() {
    return {};
}

struct Iter_less_val {
    constexpr Iter_less_val() = default;

    explicit Iter_less_val(Iter_less_iter) {}

    template<typename Iterator, typename Value>
    bool operator()(Iterator it, Value& val) const {
        return *it < val;
    }
};

inline Iter_less_val iter_less_val() {
    return {};
}

inline Iter_less_val iter_comp_val(Iter_less_iter) {
    return iter_less_val();
}

struct Val_less_iter {
    constexpr Val_less_iter() = default;
    explicit Val_less_iter(Iter_less_iter) {}

    template<typename Value, typename Iterator>
    bool operator()(Value& val, Iterator it) const {
        return val < *it;
    }
};

inline Val_less_iter val_less_iter() {
    return {};
}

inline Val_less_iter val_comp_iter(Iter_less_iter) {
    return val_less_iter();
}

struct Iter_equal_to_iter {
    template<typename Iterator1, typename Iterator2>
    bool operator()(Iterator1 it1, Iterator2 it2) const {
        return *it1 == *it2;
    }
};

inline Iter_equal_to_iter iter_equal_to_iter() {
    return {};
}

struct Iter_equal_to_val {
    template<typename Iterator, typename Value>
    bool operator()(Iterator it, Value val) const {
        return *it == val;
    }
};

inline Iter_equal_to_val iter_equal_to_val() {
    return {};
}

inline Iter_equal_to_val iter_comp_val(Iter_equal_to_iter) {
    return iter_equal_to_val();
}

template<typename Compare>
struct Iter_comp_iter {
    Compare m_comp;

    explicit Iter_comp_iter(Compare comp) : m_comp(std::move(comp)) {}

    template<typename Iterator1, typename Iterator2>
    constexpr bool operator()(Iterator1 it1, Iterator2 it2) const {
        return static_cast<bool>(m_comp(*it1, *it2));
    }
};

template<typename Compare>
constexpr Iter_comp_iter<Compare> iter_comp_iter(Compare comp) {
    return Iter_comp_iter<Compare>(std::move(comp));
}

template<typename Compare>
struct Iter_comp_val {
    Compare m_comp;

    explicit Iter_comp_val(Compare comp) : m_comp(std::move(comp)) {}

    explicit Iter_comp_val(const Iter_comp_iter<Compare>& comp) : m_comp(comp.m_comp) {}

    explicit Iter_comp_val(Iter_comp_iter<Compare>&& comp) : m_comp(std::move(comp.m_comp)) {}

    template<typename Iterator, typename Value>
    bool operator()(Iterator it, Value& val) {
        return static_cast<bool>(m_comp(*it, val));
    }
};

template<typename Compare>
constexpr Iter_comp_val<Compare> iter_comp_val(Compare comp) {
    return Iter_comp_val<Compare>(std::move(comp));
}

template<typename Compare>
constexpr Iter_comp_val<Compare> iter_comp_val(Iter_comp_iter<Compare> comp) {
    return Iter_comp_val<Compare>(std::move(comp));
}

template<typename Compare>
struct Val_comp_iter {
    Compare m_comp;

    explicit Val_comp_iter(Compare comp) : m_comp(std::move(comp)) {}
    explicit Val_comp_iter(const Iter_comp_iter<Compare>& comp) : m_comp(comp.m_comp) {}
    explicit Val_comp_iter(Iter_comp_iter<Compare>&& comp) : m_comp(std::move(comp.m_comp)) {}

    template<typename Value, typename Iterator>
    bool operator()(Value& val, Iterator it) {
        return static_cast<bool>(m_comp(val, *it));
    }
};

template<typename Compare>
constexpr Val_comp_iter<Compare> val_comp_iter(Compare comp) {
    return Val_comp_iter<Compare>(std::move(comp));
}

template<typename Compare>
constexpr Val_comp_iter<Compare> val_comp_iter(Iter_comp_iter<Compare> comp) {
    return Val_comp_iter<Compare>(std::move(comp));
}

template<typename Value>
struct Iter_equals_val {
    Value& m_value;

    explicit Iter_equals_val(Value& value) : m_value(value) {}

    template<typename Iterator>
    bool operator()(Iterator it) {
        return *it == m_value;
    }
};

template<typename Value>
constexpr Iter_equals_val<Value> iter_equals_val(Value& val) {
    return Iter_equals_val<Value>(val);
}

template<typename Iterator1>
struct Iter_equals_iter {
    Iterator1 m_it1;

    explicit Iter_equals_iter(Iterator1 it1) : m_it1(it1) {}

    template<typename Iterator2>
    bool operator()(Iterator2 it2) {
        return *it2 == *m_it1;
    }
};

template<typename Iterator>
constexpr Iter_equals_iter<Iterator> iter_comp_iter(Iter_equal_to_iter, Iterator it) {
    return Iter_equals_iter<Iterator>(it);
}


}// namespace atp

#endif//ATL_COMPARE_OPS_H
