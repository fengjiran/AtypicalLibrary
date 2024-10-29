//
// Created by richard on 10/25/24.
//

#ifndef ATL_ALLOCATOR_TRAITS_H
#define ATL_ALLOCATOR_TRAITS_H

#include "atl_type_traits.h"
#include "atl_pointer_traits.h"

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


// void pointer


template<typename alloc>
struct allocator_traits {
    using allocator_type = alloc;
};

}// namespace atp

#endif//ATL_ALLOCATOR_TRAITS_H
