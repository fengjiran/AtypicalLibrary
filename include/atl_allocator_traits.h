//
// Created by richard on 10/25/24.
//

#ifndef ATL_ALLOCATOR_TRAITS_H
#define ATL_ALLOCATOR_TRAITS_H

#include "atl_pointer_traits.h"
#include "atl_type_traits.h"

namespace atp {

#define ATL_ALLOCATOR_TRAITS_HAS_XXX(NAME, PROPERTY) \
    template<typename alloc, typename = void>        \
    struct NAME : false_type {};                     \
    template<typename alloc>                         \
    struct NAME<alloc, void_t<typename alloc::PROPERTY>> : true_type {}

// has pointer
ATL_ALLOCATOR_TRAITS_HAS_XXX(has_pointer, pointer);
template<typename T,
         typename alloc,
         typename raw_alloc = remove_reference_t<alloc>,
         bool = has_pointer<raw_alloc>::value>
struct pointer {
    using type = typename raw_alloc::pointer;
};

template<typename T,
         typename alloc,
         typename raw_alloc>
struct pointer<T, alloc, raw_alloc, false> {
    using type = T*;
};

// const pointer
ATL_ALLOCATOR_TRAITS_HAS_XXX(has_const_pointer, const_pointer);
template<typename T, typename Ptr, typename alloc, bool = has_const_pointer<alloc>::value>
struct const_pointer {
    using type = typename alloc::const_pointer;
};

template<typename T, typename Ptr, typename alloc>
struct const_pointer<T, Ptr, alloc, false> {
    using type = typename pointer_traits<Ptr>::template rebind<const T>;
};

// void pointer
ATL_ALLOCATOR_TRAITS_HAS_XXX(has_void_pointer, void_pointer);
template<typename Ptr, typename alloc, bool = has_void_pointer<alloc>::value>
struct void_pointer {
    using type = typename alloc::void_pointer;
};

template<typename Ptr, typename alloc>
struct void_pointer<Ptr, alloc, false> {
    using type = typename pointer_traits<Ptr>::template rebind<void>;
};


template<typename alloc>
struct allocator_traits {
    using allocator_type = alloc;
    using value_type = typename allocator_type::value_type;
    using pointer = typename pointer<value_type, allocator_type>::type;
};

}// namespace atp

#endif//ATL_ALLOCATOR_TRAITS_H
