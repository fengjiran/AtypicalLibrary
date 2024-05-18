//
// Created by richard on 5/4/24.
//

#include "atl_vector.hpp"
#include "gtest/gtest.h"
#include <list>

namespace atp {

TEST(MyVectorTest, ctor) {
    vec<std::string> words1{"the", "yogurt", "is", "also", "cursed"};
    std::cout << "1: " << words1;

    vec<std::string> words2(words1.begin(), words1.end());
    std::cout << "2: " << words2;

    vec<std::string> words3(words1);
    std::cout << "3: " << words3;

    vec<std::string> words4(5, "Mo");
    std::cout << "4: " << words4;

    auto const rg = {"cat", "cow", "crow"};
#ifdef __cpp_lib_containers_ranges
    vec<std::string> words5(std::from_range, rg);// overload (11)
#else
    vec<std::string> words5(rg.begin(), rg.end());// overload (5)
#endif
    std::cout << "5: " << words5;

    std::list<int> lst = {1, 2, 3, 4, 5};
    vec<int> words6(lst.begin(), lst.end());
    std::cout << "6: " << words6;

    vec<std::string> words7;
    words7 = words1;
    std::cout << "7: " << words7;

    vec<std::string> words8(std::move(words7));
    std::cout << "8: " << words8;

    auto alloc1 = std::allocator<int>();
    auto alloc2 = MyAllocator<int>();
    std::vector<int> a(alloc1);
    vec<int, std::allocator<int>> b(alloc1);

    std::vector<int, MyAllocator<int>> aa(alloc2);
    vec<int, MyAllocator<int>> bb(alloc2);

    ASSERT_TRUE(a.empty());
    ASSERT_TRUE(b.empty());
    ASSERT_TRUE(aa.empty());
    ASSERT_TRUE(bb.empty());
}

TEST(MyVectorTest, general) {
    vec<int> v{8, 4, 5, 9};
    v.push_back(6);
    v.push_back(9);
    v[2] = -1;

    vec<int> ans{8, 4, -1, 9, 6, 9};
    ASSERT_TRUE(v == ans);
}

TEST(MyVectorTest, assignment) {
    auto print = [](auto const comment, auto const& container) {
        auto size = std::size(container);
        std::cout << comment << "{ ";
        for (auto const& element: container)
            std::cout << element << (--size ? ", " : " ");
        std::cout << "}\n";
    };

    vec<int> x{1, 2, 3}, y, z;
    const auto w = {4, 5, 6, 7};

    std::cout << "Initially:\n";
    print("x = ", x);
    print("y = ", y);
    print("z = ", z);

    std::cout << "Copy assignment copies data from x to y:\n";
    y = x;
    print("x = ", x);
    print("y = ", y);

    std::cout << "Move assignment moves data from x to z, modifying both x and z:\n";
    z = std::move(x);
    print("x = ", x);
    print("z = ", z);

    std::cout << "Assignment of initializer_list w to z:\n";
    z = w;
    print("w = ", w);
    print("z = ", z);
}

TEST(MyVectorTest, back) {
    vec<char> letters{'a', 'b', 'c', 'd', 'e', 'f'};
    std::cout << letters;
    ASSERT_EQ(letters.back(), 'f');
    if (!letters.empty())
        std::cout << "The last character is '" << letters.back() << "'.\n";
}

TEST(MyVectorTest, integral_constant) {
    static_assert(true_type::value);
    static_assert(!false_type::value);

    class A {
    public:
        A() = default;
        static A select_on_container_copy_construction() { return {}; }
    };

    static_assert(_has_select_on_container_copy_construction<A>::value);
    static_assert(!_has_select_on_container_copy_construction<std::allocator<int>>::value);
    static_assert(_has_select_on_container_copy_construction<MyAllocator<int>>::value);
}

}// namespace XAcceleratorEngine
