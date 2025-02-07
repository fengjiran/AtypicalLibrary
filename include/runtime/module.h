//
// Created by richard on 2/4/25.
//

#ifndef MODULE_H
#define MODULE_H

#include "runtime/c_runtime_api.h"
#include "runtime/object.h"
#include "runtime/string.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace litetvm::runtime {

/*!
 * \brief Property of runtime module
 * We classify the property of runtime module into the following categories.
 */
enum class ModulePropertyMask : int {
    /*! \brief kBinarySerializable
   *  we can serialize the module to the stream of bytes. CUDA/OpenCL/JSON
   * runtime are representative examples. A binary exportable module can be integrated into final
   * runtime artifact by being serialized as data into the artifact, then deserialized at runtime.
   * This class of modules must implement SaveToBinary, and have a matching deserializer registered
   * as 'runtime.module.loadbinary_<type_key>'.
   */
    kBinarySerializable = 0b001,
    /*! \brief kRunnable
   * we can run the module directly. LLVM/CUDA/JSON runtime, executors (e.g,
   * virtual machine) runtimes are runnable. Non-runnable modules, such as CSourceModule, requires a
   * few extra steps (e.g,. compilation, link) to make it runnable.
   */
    kRunnable = 0b010,
    /*! \brief kDSOExportable
   * we can export the module as DSO. A DSO exportable module (e.g., a
   * CSourceModuleNode of type_key 'c') can be incorporated into the final runtime artifact (ie
   * shared library) by compilation and/or linking using the external compiler (llvm, nvcc, etc).
   * DSO exportable modules must implement SaveToFile. In general, DSO exportable modules are not
   * runnable unless there is a special support like JIT for `LLVMModule`.
   */
    kDSOExportable = 0b100
};

class ModuleNode;
class PackedFunc;

/*!
 * \brief Module container of TVM.
 */
class Module : public ObjectRef {
public:
    Module() {}
    // constructor from container.
    explicit Module(ObjectPtr<Object> n) : ObjectRef(n) {}
    /*!
   * \brief Get packed function from current module by name.
   *
   * \param name The name of the function.
   * \param query_imports Whether also query dependency modules.
   * \return The result function.
   *  This function will return PackedFunc(nullptr) if function do not exist.
   * \note Implemented in packed_func.cc
   */
    inline PackedFunc GetFunction(const String& name, bool query_imports = false);
    // The following functions requires link with runtime.
    /*!
   * \brief Import another module into this module.
   * \param other The module to be imported.
   *
   * \note Cyclic dependency is not allowed among modules,
   *  An error will be thrown when cyclic dependency is detected.
   */
    inline void Import(Module other);
    /*! \return internal container */
    inline ModuleNode* operator->();
    /*! \return internal container */
    inline const ModuleNode* operator->() const;
    /*!
   * \brief Load a module from file.
   * \param file_name The name of the host function module.
   * \param format The format of the file.
   * \note This function won't load the import relationship.
   *  Re-create import relationship by calling Import.
   */
    static Module LoadFromFile(const String& file_name, const String& format = "");
    // refer to the corresponding container.
    using ContainerType = ModuleNode;
    friend class ModuleNode;
};

/*!
 * \brief Base container of module.
 *
 * Please subclass ModuleNode to create a specific runtime module.
 *
 * \code
 *
 *  class MyModuleNode : public ModuleNode {
 *   public:
 *    // implement the interface
 *  };
 *
 *  // use make_object to create a specific
 *  // instace of MyModuleNode.
 *  Module CreateMyModule() {
 *    ObjectPtr<MyModuleNode> n =
 *      tvm::runtime::make_object<MyModuleNode>();
 *    return Module(n);
 *  }
 *
 * \endcode
 */
class ModuleNode : public Object {
public:
    /*! \brief virtual destructor */
    virtual ~ModuleNode() = default;
    /*!
   * \return The per module type key.
   * \note This key is used to for serializing custom modules.
   */
    virtual const char* type_key() const = 0;
    /*!
   * \brief Get a PackedFunc from module.
   *
   *  The PackedFunc may not be fully initialized,
   *  there might still be first time running overhead when
   *  executing the function on certain devices.
   *  For benchmarking, use prepare to eliminate
   *
   * \param name the name of the function.
   * \param sptr_to_self The ObjectPtr that points to this module node.
   *
   * \return PackedFunc(nullptr) when it is not available.
   *
   * \note The function will always remain valid.
   *   If the function need resource from the module(e.g. late linking),
   *   it should capture sptr_to_self.
   */
    virtual PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) = 0;
    /*!
   * \brief Save the module to file.
   * \param file_name The file to be saved to.
   * \param format The format of the file.
   */
    virtual void SaveToFile(const String& file_name, const String& format);
    /*!
   * \brief Save the module to binary stream.
   * \param stream The binary stream to save to.
   * \note It is recommended to implement this for device modules,
   *   but not necessarily host modules.
   *   We can use this to do AOT loading of bundled device functions.
   */
    // virtual void SaveToBinary(dmlc::Stream* stream);
    /*!
   * \brief Get the source code of module, when available.
   * \param format Format of the source code, can be empty by default.
   * \return Possible source code when available.
   */
    virtual String GetSource(const String& format = "");
    /*!
   * \brief Get the format of the module, when available.
   * \return Possible format when available.
   */
    virtual String GetFormat();
    /*!
   * \brief Get packed function from current module by name.
   *
   * \param name The name of the function.
   * \param query_imports Whether also query dependency modules.
   * \return The result function.
   *  This function will return PackedFunc(nullptr) if function do not exist.
   * \note Implemented in packed_func.cc
   */
    PackedFunc GetFunction(const String& name, bool query_imports = false);
    /*!
   * \brief Import another module into this module.
   * \param other The module to be imported.
   *
   * \note Cyclic dependency is not allowed among modules,
   *  An error will be thrown when cyclic dependency is detected.
   */
    void Import(Module other);
    /*!
   * \brief Get a function from current environment
   *  The environment includes all the imports as well as Global functions.
   *
   * \param name name of the function.
   * \return The corresponding function.
   */
    const PackedFunc* GetFuncFromEnv(const String& name);

    /*! \brief Clear all imports of the module. */
    void ClearImports() { imports_.clear(); }

    /*! \return The module it imports from */
    const std::vector<Module>& imports() const { return imports_; }

    /*!
   * \brief Returns bitmap of property.
   * By default, none of the property is set. Derived class can override this function and set its
   * own property.
   */
    virtual int GetPropertyMask() const { return 0b000; }

    /*! \brief Returns true if this module is 'DSO exportable'. */
    bool IsDSOExportable() const {
        return (GetPropertyMask() & static_cast<int>(ModulePropertyMask::kDSOExportable)) != 0;
    }

    /*! \brief Returns true if this module is 'Binary Serializable'. */
    bool IsBinarySerializable() const {
        return (GetPropertyMask() & static_cast<int>(ModulePropertyMask::kBinarySerializable)) != 0;
    }

    /*!
   * \brief Returns true if this module has a definition for a function of \p name. If
   * \p query_imports is true, also search in any imported modules.
   *
   * Note that even if this function returns true the corresponding \p GetFunction result may be
   * nullptr if the function is not yet callable without further compilation.
   *
   * The default implementation just checkis if \p GetFunction is non-null.
   */
    virtual bool ImplementsFunction(const String& name, bool query_imports = false);

    // integration with the existing components.
    static constexpr uint32_t _type_index = static_cast<uint32_t>(TypeIndex::kRuntimeModule);
    static constexpr const char* _type_key = "runtime.Module";
    // NOTE: ModuleNode can still be sub-classed
    //
    TVM_DECLARE_FINAL_OBJECT_INFO(ModuleNode, Object);

protected:
    friend class Module;
    friend class ModuleInternal;
    /*! \brief The modules this module depend on */
    std::vector<Module> imports_;

private:
    /*! \brief Cache used by GetImport */
    std::unordered_map<std::string, std::shared_ptr<PackedFunc>> import_cache_;
    std::mutex mutex_;
};

inline void Module::Import(Module other) {
    return (*this)->Import(other);
}

inline ModuleNode* Module::operator->() {
    return static_cast<ModuleNode*>(get_mutable());
}

inline const ModuleNode* Module::operator->() const {
    return static_cast<const ModuleNode*>(get());
}

inline std::ostream& operator<<(std::ostream& out, const Module& module) {
    out << "Module(type_key= ";
    out << module->type_key();
    out << ")";

    return out;
}

}// namespace litetvm::runtime

#endif//MODULE_H
