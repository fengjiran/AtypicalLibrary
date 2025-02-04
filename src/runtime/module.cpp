//
// Created by richard on 2/4/25.
//

#include "runtime/module.h"

#include <unordered_set>

namespace litetvm::runtime {

void ModuleNode::Import(Module other) {
    // specially handle rpc
    if (!std::strcmp(this->type_key(), "rpc")) {
        static const PackedFunc* fimport_ = nullptr;
        if (fimport_ == nullptr) {
            fimport_ = runtime::Registry::Get("rpc.ImportRemoteModule");
            CHECK(fimport_ != nullptr);
        }
        (*fimport_)(GetRef<Module>(this), other);
        return;
    }
    // cyclic detection.
    std::unordered_set<const ModuleNode*> visited{other.operator->()};
    std::vector<const ModuleNode*> stack{other.operator->()};
    while (!stack.empty()) {
        const ModuleNode* n = stack.back();
        stack.pop_back();
        for (const Module& m : n->imports_) {
            const ModuleNode* next = m.operator->();
            if (visited.count(next)) continue;
            visited.insert(next);
            stack.push_back(next);
        }
    }
    CHECK(!visited.count(this)) << "Cyclic dependency detected during import";
    this->imports_.emplace_back(std::move(other));
}

String ModuleNode::GetFormat() {
    LOG(FATAL) << "Module[" << type_key() << "] does not support GetFormat";
}

}
