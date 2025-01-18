//
// Created by richard on 7/3/24.
//

#include "object.h"
#include "data_type.h"

#include <string>
#include <vector>
#include <unordered_map>

namespace litetvm::runtime {

/**
 * @brief Type information.
 */
struct TypeInfo {
    /// \brief name of the type.
    std::string name;

    /// \brief hash of the name.
    size_t nameHash{0};

    /// \brief The current type index in type table.
    uint32_t index{0};

    /// \brief Parent index in type table.
    uint32_t parentIndex{0};

    /// \brief Total number of slots reserved for the type and its children.
    uint32_t numSlots{0};

    uint32_t allocatedSlots{0};

    bool childSlotsCanOverflow{true};
};

}// namespace litetvm::runtime