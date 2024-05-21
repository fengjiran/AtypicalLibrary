//
// Created by richard on 5/4/24.
//

#include "atl_vector.hpp"
#include "gtest/gtest.h"
#include <list>

TEST(ATLVectorTest, ctor) {
    atp::vector<std::string> words1{"the", "yogurt", "is", "also", "cursed"};
    std::cout << "1: " << words1;

    atp::vector<std::string> words2(words1.begin(), words1.end());
    std::cout << "2: " << words2;

    atp::vector<std::string> words3(words1);
    std::cout << "3: " << words3;

    atp::vector<std::string> words4(5, "Mo");
    std::cout << "4: " << words4;

    auto const rg = {"cat", "cow", "crow"};
#ifdef __cpp_lib_containers_ranges
    vector<std::string> words5(std::from_range, rg);// overload (11)
#else
    atp::vector<std::string> words5(rg.begin(), rg.end());// overload (5)
#endif
    std::cout << "5: " << words5;

    std::list<int> lst = {1, 2, 3, 4, 5};
    atp::vector<int> words6(lst.begin(), lst.end());
    std::cout << "6: " << words6;

    atp::vector<std::string> words7;
    words7 = words1;
    std::cout << "7: " << words7;

    atp::vector<std::string> words8(std::move(words7));
    std::cout << "8: " << words8;

    auto alloc1 = std::allocator<int>();
    auto alloc2 = atp::ATLAllocator<int>();
    std::vector<int> a(alloc1);
    atp::vector<int, std::allocator<int>> b(alloc1);

    std::vector<int, atp::ATLAllocator<int>> aa(alloc2);
    atp::vector<int, atp::ATLAllocator<int>> bb(alloc2);

    ASSERT_TRUE(a.empty());
    ASSERT_TRUE(b.empty());
    ASSERT_TRUE(aa.empty());
    ASSERT_TRUE(bb.empty());
}

TEST(ATLVectorTest, general) {
    atp::vector<int> v{8, 4, 5, 9};
    v.push_back(6);
    v.push_back(9);
    v[2] = -1;

    atp::vector<int> ans{8, 4, -1, 9, 6, 9};
    ASSERT_TRUE(v == ans);
}

TEST(ATLVectorTest, assignment) {
    auto print = [](auto const comment, auto const& container) {
        auto size = std::size(container);
        std::cout << comment << "{ ";
        for (auto const& element: container)
            std::cout << element << (--size ? ", " : " ");
        std::cout << "}\n";
    };

    atp::vector<int> x{1, 2, 3}, y, z;
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

    atp::vector<char> characters;
    characters.assign(5, 'a');
    print("characters = ", characters);

    const std::string extra(6, 'b');
    characters.assign(extra.begin(), extra.end());
    print("characters = ", characters);

    //    characters.assign({'C', '+', '+', '1', '1'});
}

TEST(ATLVectorTest, back) {
    atp::vector<char> letters{'a', 'b', 'c', 'd', 'e', 'f'};
    std::cout << letters;
    ASSERT_EQ(letters.back(), 'f');
    if (!letters.empty())
        std::cout << "The last character is '" << letters.back() << "'.\n";
}

TEST(ATLVectorTest, integral_constant) {
    static_assert(atp::true_type::value);
    static_assert(!atp::false_type::value);

    class A {
    public:
        A() = default;
        static A select_on_container_copy_construction() { return {}; }
    };

    static_assert(atp::_has_select_on_container_copy_construction<A>::value);
    static_assert(!atp::_has_select_on_container_copy_construction<std::allocator<int>>::value);
    static_assert(atp::_has_select_on_container_copy_construction<atp::ATLAllocator<int>>::value);
}
