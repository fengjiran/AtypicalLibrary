//
// Created by richard on 7/3/24.
//

#ifndef ATYPICALLIBRARY_OBJECT_H
#define ATYPICALLIBRARY_OBJECT_H

#include <cstdint>
#include <string>

/*!
 * \brief Whether or not use atomic reference counter.
 *  If the reference counter is not atomic,
 *  an object cannot be owned by multiple threads.
 *  We can, however, move an object across threads
 */
#ifndef TVM_OBJECT_ATOMIC_REF_COUNTER
#define TVM_OBJECT_ATOMIC_REF_COUNTER 1
#endif

#if TVM_OBJECT_ATOMIC_REF_COUNTER
#include <atomic>
#endif// TVM_OBJECT_ATOMIC_REF_COUNTER

// nested namespace
namespace litetvm::runtime {

/**
 * @brief Namespace for the list of type index
 */
enum class TypeIndex {
    /// \brief Root object type.
    kRoot = 0,

    /// \brief runtime::Module.
    /// <p>
    /// Standard static index assignment,
    /// frontend can take benefit of these constants.
    kRuntimeModule = 1,

    /// \brief runtime::NDArray.
    kRuntimeNDArray = 2,

    /// \brief runtime::String
    kRuntimeString = 3,

    /// \brief runtime::Array.
    kRuntimeArray = 4,

    /// \brief runtime::Map.
    kRuntimeMap = 5,

    /// \brief runtime::ShapeTuple.
    kRuntimeShapeTuple = 6,

    /// \brief runtime::PackedFunc.
    kRuntimePackedFunc = 7,

    /// \brief runtime::DRef for disco distributed runtime.
    kRuntimeDiscoDRef = 8,

    /// \brief runtime::RPCObjectRef.
    kRuntimeRPCObjectRef = 9,

    /// \brief static assignments that may subject to change.
    kRuntimeClosure,
    kRuntimeADT,
    kStaticIndexEnd,

    /// \brief Type index is allocated during runtime.
    kDynamic = kStaticIndexEnd
};

class Object {
public:
    /*!
   * \brief Object deleter
   * \param self pointer to the Object.
   */
    using FDeleter = void (*)(Object* self);

    // default construct.
    Object() = default;

    // Override the copy and assign constructors to do nothing.
    // This is to make sure only contents, but not deleter and ref_counter
    // are copied when a child class copies itself.
    // This will enable us to use make_object<ObjectClass>(*obj_ptr)
    // to copy an existing object.
    Object(const Object&) {}

    Object(Object&&) noexcept {}

    Object& operator=(const Object&) {
        return *this;
    }

    Object& operator=(Object&&) noexcept {
        return *this;
    }

    /*! \return The internal runtime type index of the object. */
    uint32_t type_index() const { return type_index_; }
    /*!
     * \return the type key of the object.
     * \note this operation is expensive, can be used for error reporting.
     */
    std::string GetTypeKey() const { return TypeIndex2Key(type_index_); }
    /*!
     * \return A hash value of the return of GetTypeKey.
     */
    size_t GetTypeKeyHash() const { return TypeIndex2KeyHash(type_index_); }

    static uint32_t RuntimeTypeIndex() {
        return static_cast<uint32_t>(TypeIndex::kRoot);
    }

    static uint32_t _GetOrAllocRuntimeTypeIndex() {
        return static_cast<uint32_t>(TypeIndex::kRoot);
    }

    /*!
   * \brief Get the type key of the corresponding index from runtime.
   * \param tindex The type index.
   * \return the result.
   */
    static std::string TypeIndex2Key(uint32_t tindex);

    /*!
   * \brief Get the type key hash of the corresponding index from runtime.
   * \param tindex The type index.
   * \return the related key-hash.
   */
    static size_t TypeIndex2KeyHash(uint32_t tindex);

    /*!
   * \brief Get the type index of the corresponding key from runtime.
   * \param key The type key.
   * \return the result.
   */
    static uint32_t TypeKey2Index(const std::string& key);

    /*!
   * Check if the object is an instance of TargetType.
   * \tparam TargetType The target type to be checked.
   * \return Whether the target type is true.
   */
    template<typename TargetType>
    bool IsInstance() const;

    /*!
   * \return Whether the cell has only one reference
   * \note We use stl style naming to be consistent with known API in shared_ptr.
   */
    bool unique() const {
        return use_count() == 1;
    }

#if TVM_OBJECT_ATOMIC_REF_COUNTER
    using RefCounterType = std::atomic<int32_t>;
#else
    using RefCounterType = int32_t;
#endif

    // Default object type properties for sub-classes
    static constexpr bool _type_final = false;
    static constexpr uint32_t _type_child_slots = 0;
    static constexpr bool _type_child_slots_can_overflow = true;

    // member information
    static constexpr bool _type_has_method_visit_attrs = true;
    static constexpr bool _type_has_method_sequal_reduce = false;
    static constexpr bool _type_has_method_shash_reduce = false;

    static constexpr const char* _type_key = "runtime.Object";

    // NOTE: the following field is not type index of Object
    // but was intended to be used by sub-classes as default value.
    // The type index of Object is TypeIndex::kRoot
    static constexpr uint32_t _type_index = static_cast<uint32_t>(TypeIndex::kDynamic);

protected:
    /*!
   * \brief Get the type index using type key.
   *
   *  When the function is first time called for a type,
   *  it will register the type to the type table in the runtime.
   *  If the static_tindex is TypeIndex::kDynamic, the function will
   *  allocate a runtime type index.
   *  Otherwise, we will populate the type table and return the static index.
   *
   * \param key the type key.
   * \param static_tindex The current _type_index field.
   *                      can be TypeIndex::kDynamic.
   * \param parent_tindex The index of the parent.
   * \param type_child_slots Number of slots reserved for its children.
   * \param type_child_slots_can_overflow Whether to allow child to overflow the slots.
   * \return The allocated type index.
   */
    static uint32_t GetOrAllocRuntimeTypeIndex(const std::string& key, uint32_t static_tindex,
                                               uint32_t parent_tindex, uint32_t type_child_slots,
                                               bool type_child_slots_can_overflow);

    // reference counter related operations
    /*! \brief developer function, increases reference counter. */
    void IncRef() {
        ref_counter_.fetch_add(1, std::memory_order_relaxed);
    }

    /*!
   * \brief developer function, decrease reference counter.
   * \note The deleter will be called when ref_counter_ becomes zero.
   */
    void DecRef() {
        if (ref_counter_.fetch_sub(1, std::memory_order_release) == 1) {
            std::atomic_thread_fence(std::memory_order_acquire);
            if (this->deleter_ != nullptr) {
                (*this->deleter_)(this);
            }
        }
    }

    // The fields of the base object cell.
    /*! \brief Type index(tag) that indicates the type of the object. */
    uint32_t type_index_{0};

    /*! \brief The internal reference counter */
    RefCounterType ref_counter_{0};

    /*!
   * \brief deleter of this object to enable customized allocation.
   * If the deleter is nullptr, no deletion will be performed.
   * The creator of the object must always set the deleter field properly.
   */
    FDeleter deleter_ = nullptr;

    // Invariant checks.
    static_assert(sizeof(int32_t) == sizeof(RefCounterType) &&
                          alignof(int32_t) == sizeof(RefCounterType),
                  "RefCounter ABI check.");

private:
    /*!
   * \return The usage count of the cell.
   * \note We use stl style naming to be consistent with known API in shared_ptr.
   */
    int use_count() const {
        return ref_counter_.load(std::memory_order_relaxed);
    }

    /*!
   * \brief Check of this object is derived from the parent.
   * \param parent_tindex The parent type index.
   * \return The derivation results.
   */
    bool DerivedFrom(uint32_t parent_tindex) const;

    template<typename>
    friend class ObjectPtr;
};

/*!
 * \brief A custom smart pointer for Object.
 * \tparam T the content data type.
 * \sa make_object
 */
template<typename T>
class ObjectPtr {
public:
    ObjectPtr() = default;

    explicit ObjectPtr(std::nullptr_t) {}

    ObjectPtr(const ObjectPtr& other) : ObjectPtr(other.data_) {}

    /*!
   * \brief copy constructor
   * \param other The value to be moved
   */
    template<typename U,
             typename = std::enable_if_t<std::is_base_of_v<T, U>>>
    explicit ObjectPtr(const ObjectPtr<U>& other) : ObjectPtr(other.data_) {}

    /*!
   * \brief move constructor
   * \param other The value to be moved
   */
    ObjectPtr(ObjectPtr&& other) noexcept : data_(other.data_) {
        other.data_ = nullptr;
    }

    /*!
   * \brief move constructor
   * \param other The value to be moved
   */
    template<typename U,
             typename = std::enable_if_t<std::is_base_of_v<T, U>>>
    explicit ObjectPtr(ObjectPtr<U>&& other) : data_(other.data_) {
        other.data_ = nullptr;
    }

    ~ObjectPtr() {
        reset();
    }

    /*!
   * \brief Swap this array with another Object
   * \param other The other Object
   */
    // void swap(ObjectPtr& other) noexcept {
    //     std::swap(data_, other.data_);
    // }

    // Get the content of the pointer
    T* get() const {
        return static_cast<T*>(data_);
    }

    T* operator->() const {
        return get();
    }

    T& operator*() const {
        return *get();
    }

    /*!
   * \brief copy assignment
   * \param other The value to be assigned.
   * \return reference to self.
   */
    ObjectPtr& operator=(const ObjectPtr& other) {
        ObjectPtr tmp(other);
        swap(tmp, *this);
        return *this;
    }

    /*!
   * \brief move assignment
   * \param other The value to be assigned.
   * \return reference to self.
   */
    ObjectPtr& operator=(ObjectPtr&& other) noexcept {
        ObjectPtr tmp(std::move(other));
        swap(tmp, *this);
        return *this;
    }

    /*!
   * \brief nullptr check
   * \return result of comparison of internal pointer with nullptr.
   */
    explicit operator bool() const {
        return get() != nullptr;
    }

    /*! \brief reset the content of ptr to be nullptr */
    void reset() {
        if (data_) {
            data_->DecRef();
            data_ = nullptr;
        }
    }

private:
    Object* data_{nullptr};

    explicit ObjectPtr(Object* data) : data_(data) {
        if (data) {
            data_->IncRef();
        }
    }

    friend void swap(ObjectPtr& a, ObjectPtr& b) noexcept {
        std::swap(a.data_, b.data_);
    }

    friend class Object;
};

template<typename TargetType>
bool Object::IsInstance() const {
    const Object* self = this;

    // Everything is a subclass of object.
    if (std::is_same_v<TargetType, Object>) {
        return true;
    }

    if (TargetType::_type_final) {
        // if the target type is a final type
        // then we only need to check the equivalence.
        return self->type_index_ == TargetType::RuntimeTypeIndex();
    } else {
        // if target type is a non-leaf type
        // Check if type index falls into the range of reserved slots.
        uint32_t begin = TargetType::RuntimeTypeIndex();

        if (TargetType::_type_child_slots != 0) {
            uint32_t end = begin + TargetType::_type_child_slots;
            if (self->type_index_ >= begin && self->type_index_ < end) {
                return true;
            }
        } else {
            if (self->type_index_ == begin) {
                return true;
            }
        }

        if (!TargetType::_type_child_slots_can_overflow) {
            return false;
        }

        if (self->type_index_ < TargetType::RuntimeTypeIndex()) {
            return false;
        }

        return self->DerivedFrom(TargetType::RuntimeTypeIndex());
    }
}

}// namespace litetvm::runtime


#endif//ATYPICALLIBRARY_OBJECT_H
