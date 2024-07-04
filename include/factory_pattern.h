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

    virtual void SetName(const std::string& name) = 0;

    virtual void SetSize(int size) = 0;
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

    void SetName(const std::string& name) override {
        name_ = name;
    }

    void SetSize(int size) override {
        size_ = size;
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

    void SetName(const std::string& name) override {
        name_ = name;
    }

    void SetSize(int size) override {
        size_ = size;
    }

private:
    std::string name_;
    int size_{};
};

template<typename Base>
class Factory {
public:
    using FCreator = std::function<std::shared_ptr<Base>()>;

    static Factory& GetInstance() {
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

template<typename Impl>
class RegisterEntry {
public:
    explicit RegisterEntry(const std::string& name) {
        Factory<Car>::GetInstance().Register(name, [] {
            return std::make_shared<Impl>();
        });
    }
};

#define CONCAT_AB_(A, B) A##B
#define CONCAT_AB(A, B) CONCAT_AB_(A, B)
#define REGISTER_CAR(type, name) static RegisterEntry<type> CONCAT_AB(reg_, __COUNTER__) = RegisterEntry<type>(name)

#endif//ATYPICALLIBRARY_FACTORY_PATTERN_H
