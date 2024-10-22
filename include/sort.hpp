#ifndef SORT_HPP
#define SORT_HPP

#include <iostream>

namespace atp {

class Sort {
public:
    template<typename iterator>
    static void Show(iterator begin, iterator end) {
        for (auto it = begin; it != end; ++it) {
            std::cout << *it << ", ";
        }
        std::cout << std::endl;
    }
};

}// namespace atp

#endif