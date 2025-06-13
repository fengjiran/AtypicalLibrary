//
// Created by 赵丹 on 25-6-10.
//

#ifndef TRIE_TREE_H
#define TRIE_TREE_H

#include <string>
#include <vector>

template<int R = 256>
class Trie {
public:
    Trie() {
        is_end = false;
        children.resize(R);
        children.shrink_to_fit();
    }

    std::vector<Trie*> children;
    bool is_end;
};


#endif//TRIE_TREE_H
