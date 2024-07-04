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
}
