//
// Created by richard on 7/3/24.
//

#include "object.h"
#include "data_type.h"

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace litetvm::runtime {

/**
 * @brief Type information.
 */
struct TypeInfo {
    /// \brief name of the type.
    std::string name;

    /// \brief hash of the name.
    size_t name_hash{0};

    /// \brief The current type index in type table.
    uint32_t index{0};

    /// \brief Parent index in type table.
    uint32_t parent_index{0};

    // NOTE: the indices in [index, index + num_reserved_slots) are
    // reserved for the child-class of this type.
    /// \brief Total number of slots reserved for the type and its children.
    uint32_t num_slots{0};

    /// \brief number of allocated child slots.
    uint32_t allocated_slots{0};

    /// \brief Whether child can overflow.
    bool child_slots_can_overflow{true};
};

class TypeContext {
public:
    static TypeContext& Global() {
        static TypeContext instance;
        return instance;
    }

    uint32_t GetOrAllocRuntimeTypeIndex(const std::string& skey, uint32_t static_tindex,
                                        uint32_t parent_tindex, uint32_t num_child_slots,
                                        bool child_slots_can_overflow) {
        std::lock_guard lock(mutex_);
        if (type_key2index_.find(skey) != type_key2index_.end()) {
            return type_key2index_[skey];
        }

        // try to allocate from parent's type table.
        CHECK_LT(parent_tindex, type_table_.size()) << " skey=" << skey << ", static_tindex=" << static_tindex;
        auto& pinfo = type_table_[parent_tindex];
        CHECK_EQ(pinfo.index, parent_tindex);

        // if parent cannot overflow, then this class cannot.
        child_slots_can_overflow = pinfo.child_slots_can_overflow && child_slots_can_overflow;

        // total number of slots include the type itself.

    }

private:
    TypeContext()
        : type_table_(static_cast<uint32_t>(TypeIndex::kStaticIndexEnd), TypeInfo()),
          mutex_({}),
          type_counter_(static_cast<uint32_t>(TypeIndex::kStaticIndexEnd)),
          type_key2index_({}) {
        type_table_[0].name = "runtime.Object";
    }

    std::vector<TypeInfo> type_table_;
    std::mutex mutex_;// mutex to avoid registration from multiple threads.
    std::atomic<uint32_t> type_counter_;
    std::unordered_map<std::string, uint32_t> type_key2index_;
};

}// namespace litetvm::runtime