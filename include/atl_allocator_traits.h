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

// const void pointer
ATL_ALLOCATOR_TRAITS_HAS_XXX(has_const_void_pointer, const_void_pointer);
template<typename Ptr, typename alloc, bool = has_const_void_pointer<alloc>::value>
struct const_void_pointer {
    using type = typename alloc::const_void_pointer;
};

template<typename Ptr, typename alloc>
struct const_void_pointer<Ptr, alloc, false> {
    using type = typename pointer_traits<Ptr>::template rebind<const void>;
};

// size type
ATL_ALLOCATOR_TRAITS_HAS_XXX(has_size_type, size_type);
template<typename alloc, typename DiffType, bool = has_size_type<alloc>::value>
struct size_type : std::make_unsigned<DiffType> {};

template<typename alloc, typename DiffType>
struct size_type<alloc, DiffType, true> {
    using type = typename alloc::size_type;
};

// difference type
ATL_ALLOCATOR_TRAITS_HAS_XXX(has_alloc_traits_difference_type, difference_type);
template<typename alloc, typename Ptr, bool = has_alloc_traits_difference_type<alloc>::value>
struct alloc_traits_difference_type {
    using type = typename pointer_traits<Ptr>::difference_type;
};

template<typename alloc, typename Ptr>
struct alloc_traits_difference_type<alloc, Ptr, true> {
    using type = typename alloc::difference_type;
};

// propagate on container copy assignment
ATL_ALLOCATOR_TRAITS_HAS_XXX(has_propagate_on_container_copy_assignment, propagate_on_container_copy_assignment);
template<typename alloc, bool = has_propagate_on_container_copy_assignment<alloc>::value>
struct propagate_on_container_copy_assignment : false_type {};

template<typename alloc>
struct propagate_on_container_copy_assignment<alloc, true> {
    using type = typename alloc::propagate_on_container_copy_assignment;
};

// propagate on container move assignment
ATL_ALLOCATOR_TRAITS_HAS_XXX(has_propagate_on_container_move_assignment, propagate_on_container_move_assignment);
template<typename alloc, bool = has_propagate_on_container_move_assignment<alloc>::value>
struct propagate_on_container_move_assignment : false_type {};

template<typename alloc>
struct propagate_on_container_move_assignment<alloc, true> {
    using type = typename alloc::propagate_on_container_move_assignment;
};

// propagate on container swap
ATL_ALLOCATOR_TRAITS_HAS_XXX(has_propagate_on_container_swap, propagate_on_container_swap);
template<typename alloc, bool = has_propagate_on_container_swap<alloc>::value>
struct propagate_on_container_swap : false_type {};

template<typename alloc>
struct propagate_on_container_swap<alloc, true> {
    using type = typename alloc::propagate_on_container_swap;
};

ATL_ALLOCATOR_TRAITS_HAS_XXX(has_is_always_true, is_always_true);
template<typename alloc, bool = has_is_always_true<alloc>::value>
struct is_always_true : std::is_empty<alloc> {};

template<typename alloc>
struct is_always_true<alloc, true> {
    using type = typename alloc::is_always_true;
};


template<typename alloc>
struct allocator_traits {
    using allocator_type = alloc;
    using value_type = typename allocator_type::value_type;
    using pointer = typename pointer<value_type, allocator_type>::type;
};

}// namespace atp

#endif//ATL_ALLOCATOR_TRAITS_H
