//
// Created by 赵丹 on 25-7-18.
//

#ifndef SIGNATURE_H
#define SIGNATURE_H

#include "tensor_utils.h"
#include <typeindex>

namespace atp {

// A CppSignature object holds RTTI information about a C++ function signature
// at runtime and can compare them or get a debug-printable name.
class Signature {
public:
    Signature(const Signature&) = default;
    Signature(Signature&&) noexcept = default;
    Signature& operator=(const Signature&) = default;
    Signature& operator=(Signature&&) noexcept = default;

    std::string name() const {
        return demangle(signature_.name());
    }

    friend bool operator==(const Signature& lhs, const Signature& rhs) {
        if (lhs.signature_ == rhs.signature_) {
            return true;
        }

        if (strcmp(lhs.signature_.name(), rhs.signature_.name()) == 0) {
            return true;
        }
        return false;
    }

private:
    explicit Signature(std::type_index signature) : signature_(std::move(signature)) {}
    std::type_index signature_;
};

inline bool operator!=(const Signature& lhs, const Signature& rhs) {
    return !(lhs == rhs);
}

}// namespace atp

#endif//SIGNATURE_H
