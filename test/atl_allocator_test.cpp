//
// Created by 赵丹 on 24-5-10.
//

#include "atl_allocator.hpp"
#include "gtest/gtest.h"

TEST(ATLAllocatorTest, general) {
    // default allocator for ints
    atp::ATLAllocator<int> alloc1;
    static_assert(atp::is_same_v<int, decltype(alloc1)::value_type>);

    int* p1 = alloc1.allocate(1);// space for one int
    alloc1.deallocate(p1, 1);    // and it is gone

    // Even those can be used through traits though, so no need
    using traits_t1 = std::allocator_traits<decltype(alloc1)>; // The matching trait
    p1 = traits_t1::allocate(alloc1, 1);
    traits_t1::construct(alloc1, p1, 7);  // construct the int
    std::cout << *p1 << '\n';
    traits_t1::deallocate(alloc1, p1, 1); // deallocate space for one int

    // default allocator for strings
    std::allocator<std::string> alloc2;
    // matching traits
    using traits_t2 = std::allocator_traits<decltype(alloc2)>;

    // Rebinding the allocator using the trait for strings gets the same type
    traits_t2::rebind_alloc<std::string> alloc_ = alloc2;

    std::string* p2 = traits_t2::allocate(alloc2, 2); // space for 2 strings

    traits_t2::construct(alloc2, p2, "foo");
    traits_t2::construct(alloc2, p2 + 1, "bar");

    std::cout << p2[0] << ' ' << p2[1] << '\n';

    traits_t2::destroy(alloc2, p2 + 1);
    traits_t2::destroy(alloc2, p2);
    traits_t2::deallocate(alloc2, p2, 2);

    static_assert(atp::is_same_v<atp::pointer_traits_element_type<std::shared_ptr<int>>::type, int>);
    // static_assert(atp::has_rebind<atp::ATLAllocator<int>, double>::value);
}
