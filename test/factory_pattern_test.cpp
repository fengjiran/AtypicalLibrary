//
// Created by richard on 7/4/24.
//

#include "factory_pattern.h"

REGISTER_CAR(Impl1, "AITO");
REGISTER_CAR(Impl2, "LUXCEED");


TEST(FactoryPatternTest, test1) {
    auto x = Factory<Car>::GetInstance().Create("AITO");
    x->SetName("M5EV");
    x->SetSize(520);
    EXPECT_EQ(x->GetName(), "M5EV");
    EXPECT_EQ(x->GetSize(), 520);

    std::vector<int> arr {10, 20, 10, 15, 12, 7, 9};
//    auto it = std::remove(arr.begin(), arr.end(), 10);
    arr.erase(std::remove(arr.begin(), arr.end(), 10), arr.end());
    for (auto xx: arr) {
        std::cout << xx << std::endl;
    }
}
