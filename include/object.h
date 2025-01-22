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

    static uint32_t RuntimeTypeIndex() {
        return static_cast<uint32_t>(TypeIndex::kRoot);
    }

    static uint32_t _GetOrAllocRuntimeTypeIndex() {
        return static_cast<uint32_t>(TypeIndex::kRoot);
    }

#if TVM_OBJECT_ATOMIC_REF_COUNTER
    using RefCounterType = std::atomic<int32_t>;
#else
    using RefCounterType = int32_t;
#endif

    static constexpr const char* _type_key = "runtime.Object";
    // Default object type properties for sub-classes
    static constexpr bool _type_final = false;
    static constexpr uint32_t _type_child_slots = 0;
    static constexpr bool _type_child_slots_can_overflow = true;

    // member information
    static constexpr bool _type_has_method_visit_attrs = true;
    static constexpr bool _type_has_method_sequal_reduce = false;
    static constexpr bool _type_has_method_shash_reduce = false;

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
};

}// namespace litetvm::runtime


#endif//ATYPICALLIBRARY_OBJECT_H
