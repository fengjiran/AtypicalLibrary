//
// Created by richard on 7/4/24.
//

#ifndef ATYPICALLIBRARY_FACTORY_PATTERN_H
#define ATYPICALLIBRARY_FACTORY_PATTERN_H

#include "gtest/gtest.h"

class Car {
public:
    virtual ~Car() = default;

    virtual std::string GetName() const = 0;

    virtual int GetSize() const = 0;
};

class Impl1 : public Car {
public:
    Impl1() = default;

    Impl1(std::string name, int size)
        : name_(std::move(name)), size_(size) {}

    std::string GetName() const override {
        return name_;
    }

    int GetSize() const override {
        return size_;
    }

private:
    std::string name_;
    int size_{};
};

class Impl2 : public Car {
public:
    Impl2() = default;

    Impl2(std::string name, int size)
        : name_(std::move(name)), size_(size) {}

    std::string GetName() const override {
        return name_;
    }

    int GetSize() const override {
        return size_;
    }

private:
    std::string name_;
    int size_{};
};

template<typename Base>
class Factory {
public:
    using FCreator = std::function<std::shared_ptr<Base>()>;
    Factory& GetInstance() {
        static Factory inst;
        return inst;
    }

    Factory& Register(const std::string& name, FCreator creator) {
        creators_[name] = creator;
        return *this;
    }

    std::shared_ptr<Base> Create(const std::string& name) {
        if (creators_.find(name) != creators_.end()) {
            return creators_[name]();
        }
        return {};
    }

private:
    Factory() = default;

    std::map<std::string, FCreator> creators_;
};

template<typename Base, typename Impl>
class RegisterEntry {
public:
    template<typename... Args>
    explicit RegisterEntry(const std::string& name, Args... args) {
        Factory<Base>& factory = Factory<Base>::GetInstance();
        factory.Register(name, [&] {
            return std::make_shared<Base>(new Impl(std::forward<Args>(args)...));
        });
    }
};

#define CONCAT_AB_(A, B) A##B
#define CONCAT_AB(A, B) CONCAT_AB_(A, B)

#endif//ATYPICALLIBRARY_FACTORY_PATTERN_H
