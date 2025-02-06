//
// Created by 赵丹 on 25-2-6.
//

#ifndef MAP_H
#define MAP_H

namespace litetvm::runtime {

class MapNode : public Object {
public:
    /*! \brief Type of the keys in the hash map */
    using key_type = ObjectRef;
    /*! \brief Type of the values in the hash map */
    using mapped_type = ObjectRef;
    /*! \brief Type of value stored in the hash map */
    using KVType = std::pair<ObjectRef, ObjectRef>;
    /*! \brief Iterator class */
    class iterator;

    static_assert(std::is_standard_layout<KVType>::value, "KVType is not standard layout");
    static_assert(sizeof(KVType) == 16 || sizeof(KVType) == 8, "sizeof(KVType) incorrect");

    static constexpr uint32_t _type_index = static_cast<uint32_t>(TypeIndex::kRuntimeMap);
    static constexpr const char* _type_key = "Map";
    TVM_DECLARE_FINAL_OBJECT_INFO(MapNode, Object);

    /*!
   * \brief Number of elements in the SmallMapNode
   * \return The result
   */
    size_t size() const {
        return size_;
    }

    /*!
   * \brief Count the number of times a key exists in the hash map
   * \param key The indexing key
   * \return The result, 0 or 1
   */
    size_t count(const key_type& key) const;

    /*!
   * \brief Index value associated with a key, throw exception if the key does not exist
   * \param key The indexing key
   * \return The const reference to the value
   */
    const mapped_type& at(const key_type& key) const;

    /*!
   * \brief Index value associated with a key, throw exception if the key does not exist
   * \param key The indexing key
   * \return The mutable reference to the value
   */
    mapped_type& at(const key_type& key);
    /*! \return begin iterator */
    iterator begin() const;
    /*! \return end iterator */
    iterator end() const;

    /*!
   * \brief Index value associated with a key
   * \param key The indexing key
   * \return The iterator of the entry associated with the key, end iterator if not exists
   */
    iterator find(const key_type& key) const;
    /*!
   * \brief Erase the entry associated with the iterator
   * \param position The iterator
   */
    void erase(const iterator& position);

    /*!
   * \brief Erase the entry associated with the key, do nothing if not exists
   * \param key The indexing key
   */
    void erase(const key_type& key) { erase(find(key)); }

    class iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type = int64_t;
        using value_type = KVType;
        using pointer = KVType*;
        using reference = KVType&;
/*! \brief Default constructor */
#if TVM_DEBUG_WITH_ABI_CHANGE
        iterator() : state_marker(0), index(0), self(nullptr) {}
#else
        iterator() : index(0), self(nullptr) {}
#endif// TVM_DEBUG_WITH_ABI_CHANGE
        /*! \brief Compare iterators */
        bool operator==(const iterator& other) const {
            // TVM_MAP_FAIL_IF_CHANGED()
            return index == other.index && self == other.self;
        }
        /*! \brief Compare iterators */
        bool operator!=(const iterator& other) const { return !(*this == other); }
        /*! \brief De-reference iterators */
        pointer operator->() const;
        /*! \brief De-reference iterators */
        reference operator*() const {
            // TVM_MAP_FAIL_IF_CHANGED()
            return *((*this).operator->());
        }
        /*! \brief Prefix self increment, e.g. ++iter */
        iterator& operator++();
        /*! \brief Prefix self decrement, e.g. --iter */
        iterator& operator--();
        /*! \brief Suffix self increment */
        iterator operator++(int) {
            // TVM_MAP_FAIL_IF_CHANGED()
            iterator copy = *this;
            ++(*this);
            return copy;
        }
        /*! \brief Suffix self decrement */
        iterator operator--(int) {
            // TVM_MAP_FAIL_IF_CHANGED()
            iterator copy = *this;
            --(*this);
            return copy;
        }

    protected:
#if TVM_DEBUG_WITH_ABI_CHANGE
        uint64_t state_marker;
        /*! \brief Construct by value */
        iterator(uint64_t index, const MapNode* self)
            : state_marker(self->state_marker), index(index), self(self) {}

#else
        iterator(uint64_t index, const MapNode* self) : index(index), self(self) {}
#endif// TVM_DEBUG_WITH_ABI_CHANGE
        /*! \brief The position on the array */
        uint64_t index;
        /*! \brief The container it points to */
        const MapNode* self;

        friend class DenseMapNode;
        friend class SmallMapNode;
    };

    /*!
   * \brief Create an empty container
   * \return The object created
   */
    static inline ObjectPtr<MapNode> Empty();

protected:
#if TVM_DEBUG_WITH_ABI_CHANGE
    uint64_t state_marker;
#endif// TVM_DEBUG_WITH_ABI_CHANGE
    /*!
   * \brief Create the map using contents from the given iterators.
   * \param first Begin of iterator
   * \param last End of iterator
   * \tparam IterType The type of iterator
   * \return ObjectPtr to the map created
   */
    template<typename IterType>
    static ObjectPtr<Object> CreateFromRange(IterType first, IterType last);
    /*!
   * \brief InsertMaybeReHash an entry into the given hash map
   * \param kv The entry to be inserted
   * \param map The pointer to the map, can be changed if re-hashing happens
   */
    static inline void InsertMaybeReHash(const KVType& kv, ObjectPtr<Object>* map);
    /*!
   * \brief Create an empty container with elements copying from another SmallMapNode
   * \param from The source container
   * \return The object created
   */
    static inline ObjectPtr<MapNode> CopyFrom(MapNode* from);
    /*! \brief number of slots minus 1 */
    uint64_t slots_;
    /*! \brief number of entries in the container */
    uint64_t size_;
    // Reference class
    template<typename, typename, typename, typename>
    friend class Map;
};

}// namespace litetvm::runtime

#endif//MAP_H
