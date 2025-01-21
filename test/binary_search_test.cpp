//
// Created by richard on 7/31/24.
//

#include "binary_search.h"
#include "utils.hpp"
#include "gtest/gtest.h"
// #include "LeakDetector.h"

class Empty {
public:
    virtual ~Empty() = default;

    virtual void f() = 0;
};

class Base1 {
public:
    virtual void f1() {
        std::cout << "Base1 f1()\n";
    }

    virtual void f2() {
        std::cout << "Base1 f2()\n";
    }

    virtual void f3() {
        std::cout << "Base1 f3()\n";
    }

    virtual void f4() = 0;
    // public:
    virtual ~Base1() = default;
};

void Base1::f4() {
    std::cout << "pure virtual Base1 f4()\n";
}


class Derived1 : public Base1 {
    // public:
    void f1() override {
        std::cout << "Derived1 f1()\n";
    }

    virtual void f4() {
        Base1::f4();
        std::cout << "Derived1 f4()\n";
    }
};

class Derived2 : public Derived1 {
public:
    void f1() override {
        std::cout << "Derived2 f1()\n";
    }

    virtual void f5() {
        std::cout << "Derived2 f()\n";
    }

    virtual inline void f6();
};

void Derived2::f6() {
}

#pragma pack(4)
struct MyStruct {
    double a;
    int b;
    int c[2];
};

class base {
public:
    virtual void f() {}
    virtual ~base() = default;
};

class derived : public base {
};

TEST(BinarySearch, t1) {
    std::vector<int> nums1{2, 4, 5, 6, 9};
    EXPECT_EQ(BinarySearch<int>::FindFirstGEItem(nums1, 0, nums1.size() - 1, 2), 0);
    EXPECT_EQ(BinarySearch<int>::FindFirstGEItem(nums1, 0, nums1.size() - 1, 1), 0);
    EXPECT_EQ(BinarySearch<int>::FindFirstGEItem(nums1, 0, nums1.size() - 1, 9), 4);
    EXPECT_EQ(BinarySearch<int>::FindFirstGEItem(nums1, 0, nums1.size() - 1, 10), -1);

    EXPECT_EQ(BinarySearch<int>::FindFirstGTItem(nums1, 0, nums1.size() - 1, 2), 1);
    EXPECT_EQ(BinarySearch<int>::FindFirstGTItem(nums1, 0, nums1.size() - 1, 1), 0);
    EXPECT_EQ(BinarySearch<int>::FindFirstGTItem(nums1, 0, nums1.size() - 1, 9), -1);

    EXPECT_EQ(BinarySearch<int>::LeftBound(nums1, 0, nums1.size() - 1, 4), 1);
    EXPECT_EQ(BinarySearch<int>::LeftBound(nums1, 0, nums1.size() - 1, 2), 0);
    EXPECT_EQ(BinarySearch<int>::LeftBound(nums1, 0, nums1.size() - 1, 9), 4);
    EXPECT_EQ(BinarySearch<int>::LeftBound(nums1, 0, nums1.size() - 1, 1), -1);
    EXPECT_EQ(BinarySearch<int>::LeftBound(nums1, 0, nums1.size() - 1, 10), -1);

    std::vector<int> nums2{3, 5, 8, 10};
    // EXPECT_EQ(BinarySearch<int>::FindFirstGEItem(nums2, 0, nums2.size() - 1, 3), 2);
    std::reverse(nums1.begin(), nums1.end());

    auto f = [](std::vector<int>& nums, int target) {
        int left = 0;
        int right = (int) nums.size();
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        if (left == nums.size()) {
            return (int) nums.size();
        }

        return left;
    };

    int x = 15;
    int right = f(nums2, x);
    int left = right - 1;
    int k = 2;
    while (k--) {
        if (left < 0) {
            right++;
        } else if (right >= nums2.size()) {
            left--;
        } else if (x - nums2[left] <= nums2[right] - x) {
            left--;
        } else {
            right++;
        }
    }

    std::cout << left + 1 << std::endl;
    std::cout << right - 1 << std::endl;
    std::vector<int> y(nums2.begin() + left + 1, nums2.begin() + right);
    for (int t: y) {
        std::cout << t << std::endl;
    }

    std::unordered_map<std::string, int> my_map;
    my_map["hello"] = 1;
    if (auto it = my_map.find("hello"); it != my_map.end()) {
        std::cout << my_map["hello"] << std::endl;
    }

    //    int a = 10;
    //    int& b = a;
    //    static_assert(std::is_same_v<decltype((a)), int&>);
}

TEST(BinarySearch, RTPolymorphism) {
    using func = void (*)();

    // auto* p1 = new Base1;
    // void** vtable1 = *reinterpret_cast<void***>(p1);
    // for (int i = 0; i < 3; ++i) {
    //     reinterpret_cast<func>(vtable1[i])();
    // }

    Base1* p2 = new Derived1;
    auto vtable2 = *reinterpret_cast<void***>(p2);
    for (int i = 0; i < 3; ++i) {
        reinterpret_cast<func>(vtable2[i])();
    }

    p2->f4();

    // delete p1;
    delete p2;

    std::cout << sizeof(Derived1) << std::endl;
    std::cout << sizeof(Empty) << std::endl;
    std::cout << sizeof(MyStruct) << std::endl;
    std::cout << sizeof(double&&) << std::endl;
    std::cout << sizeof(derived) << std::endl;

    const int c = 10;
    int* p = const_cast<int*>(&c);
    *p = 20;
    std::cout << c << std::endl;
}
