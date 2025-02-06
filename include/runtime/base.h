//
// Created by richard on 2/4/25.
//

#ifndef BASE_H
#define BASE_H

#include "runtime/memory.h"
#include "runtime/object.h"

namespace litetvm::runtime {

/*! \brief String-aware ObjectRef equal functor */
struct ObjectHash {
    /*!
   * \brief Calculate the hash code of an ObjectRef
   * \param a The given ObjectRef
   * \return Hash code of a, string hash for strings and pointer address otherwise.
   */
    size_t operator()(const ObjectRef& a) const;
};

/*! \brief String-aware ObjectRef hash functor */
struct ObjectEqual {
    /*!
   * \brief Check if the two ObjectRef are equal
   * \param a One ObjectRef
   * \param b The other ObjectRef
   * \return String equality if both are strings, pointer address equality otherwise.
   */
    bool operator()(const ObjectRef& a, const ObjectRef& b) const;
};


/*!
 * \brief Base template for classes with array like memory layout.
 *
 *        It provides general methods to access the memory. The memory
 *        layout is ArrayType + [ElemType]. The alignment of ArrayType
 *        and ElemType is handled by the memory allocator.
 *
 * \tparam ArrayType The array header type, contains object specific metadata.
 * \tparam ElemType The type of objects stored in the array right after
 * ArrayType.
 *
 * \code
 * // Example usage of the template to define a simple array wrapper
 * class ArrayObj : public InplaceArrayBase<ArrayObj, Elem> {
 * public:
 *  // Wrap EmplaceInit to initialize the elements
 *  template <typename Iterator>
 *  void Init(Iterator begin, Iterator end) {
 *   size_t num_elems = std::distance(begin, end);
 *   auto it = begin;
 *   this->size = 0;
 *   for (size_t i = 0; i < num_elems; ++i) {
 *     InplaceArrayBase::EmplaceInit(i, *it++);
 *     this->size++;
 *   }
 *  }
 * }
 *
 * void test_function() {
 *   vector<Elem> fields;
 *   auto ptr = make_inplace_array_object<ArrayObj, Elem>(fields.size());
 *   ptr->Init(fields.begin(), fields.end());
 *
 *   // Access the 0th element in the array.
 *   assert(ptr->operator[](0) == fields[0]);
 * }
 *
 * \endcode
 */
template<typename ArrayType, typename ElemType>
class InplaceArrayBase {
public:
    /*!
   * \brief Access element at index
   * \param idx The index of the element.
   * \return Const reference to ElemType at the index.
   */
    const ElemType& operator[](size_t idx) const {
        size_t size = Self()->GetSize();
        CHECK_LT(idx, size) << "Index " << idx << " out of bounds " << size << "\n";
        return *(reinterpret_cast<ElemType*>(AddressOf(idx)));
    }

    /*!
   * \brief Access element at index
   * \param idx The index of the element.
   * \return Reference to ElemType at the index.
   */
    ElemType& operator[](size_t idx) {
        size_t size = Self()->GetSize();
        CHECK_LT(idx, size) << "Index " << idx << " out of bounds " << size << "\n";
        return *(reinterpret_cast<ElemType*>(AddressOf(idx)));
    }

    /*!
   * \brief Destroy the Inplace Array Base object
   */
    ~InplaceArrayBase() {
        if (!(std::is_standard_layout<ElemType>::value && std::is_trivial<ElemType>::value)) {
            size_t size = Self()->GetSize();
            for (size_t i = 0; i < size; ++i) {
                auto* fp = reinterpret_cast<ElemType*>(AddressOf(i));
                fp->ElemType::~ElemType();
            }
        }
    }

protected:
    /*!
   * \brief Construct a value in place with the arguments.
   *
   * \tparam Args Type parameters of the arguments.
   * \param idx Index of the element.
   * \param args Arguments to construct the new value.
   *
   * \note Please make sure ArrayType::GetSize returns 0 before first call of
   * EmplaceInit, and increment GetSize by 1 each time EmplaceInit succeeds.
   */
    template<typename... Args>
    void EmplaceInit(size_t idx, Args&&... args) {
        void* field_ptr = AddressOf(idx);
        new (field_ptr) ElemType(std::forward<Args>(args)...);
    }

    /*!
   * \brief Return the self object for the array.
   *
   * \return Pointer to ArrayType.
   */
    ArrayType* Self() const {
        return static_cast<ArrayType*>(const_cast<InplaceArrayBase*>(this));
    }

    /*!
   * \brief Return the raw pointer to the element at idx.
   *
   * \param idx The index of the element.
   * \return Raw pointer to the element.
   */
    NODISCARD void* AddressOf(size_t idx) const {
        static_assert(
                alignof(ArrayType) % alignof(ElemType) == 0 && sizeof(ArrayType) % alignof(ElemType) == 0,
                "The size and alignment of ArrayType should respect "
                "ElemType's alignment.");

        size_t kDataStart = sizeof(ArrayType);
        ArrayType* self = Self();
        char* data_start = reinterpret_cast<char*>(self) + kDataStart;
        return data_start + idx * sizeof(ElemType);
    }
};

}// namespace litetvm::runtime

#endif//BASE_H
