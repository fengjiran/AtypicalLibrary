//
// Created by richard on 7/3/24.
//

#include "object.h"
#include "data_type.h"
#include "type_context.h"

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace litetvm::runtime {

uint32_t Object::GetOrAllocRuntimeTypeIndex(const std::string& key, uint32_t static_tindex,
                                            uint32_t parent_tindex, uint32_t type_child_slots,
                                            bool type_child_slots_can_overflow) {
    return TypeContext::Global().GetOrAllocRuntimeTypeIndex(
            key, static_tindex, parent_tindex, type_child_slots, type_child_slots_can_overflow);
}


}// namespace litetvm::runtime