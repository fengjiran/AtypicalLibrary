//
// Created by richard on 2/1/25.
//

#ifndef DEVICE_API_H
#define DEVICE_API_H

#include "runtime/c_runtime_api.h"
#include "runtime/ndarray.h"
#include "runtime/packed_func.h"

#include <string>

namespace litetvm::runtime {

/*!
 * \brief the query type into GetAttr
 */
enum class DeviceAttrKind : int {
    kExist = 0,
    kMaxThreadsPerBlock = 1,
    kWarpSize = 2,
    kMaxSharedMemoryPerBlock = 3,
    kComputeVersion = 4,
    kDeviceName = 5,
    kMaxClockRate = 6,
    kMultiProcessorCount = 7,
    kMaxThreadDimensions = 8,
    kMaxRegistersPerBlock = 9,
    kGcnArch = 10,
    kApiVersion = 11,
    kDriverVersion = 12,
    kL2CacheSizeBytes = 13,
    kTotalGlobalMemory = 14,
    kAvailableGlobalMemory = 15,
};

#ifdef TVM_KALLOC_ALIGNMENT
/*! \brief Number of bytes each allocation must align to */
constexpr int kAllocAlignment = TVM_KALLOC_ALIGNMENT;

/*! \brief Number of bytes each allocation must align to in temporary allocation */
constexpr int kTempAllocaAlignment = TVM_KALLOC_ALIGNMENT;
#else
/*! \brief Number of bytes each allocation must align to */
constexpr int kAllocAlignment = 64;

/*! \brief Number of bytes each allocation must align to in temporary allocation */
constexpr int kTempAllocaAlignment = 64;
#endif// TVM_KALLOC_ALIGNMENT

/*! \brief Maximum size that can be allocated on stack */
constexpr int kMaxStackAlloca = 1024;

/*! \brief Number of bytes each allocation must align to by default in the workspace buffer to
 * service intermediate tensors */
constexpr int kDefaultWorkspaceAlignment = 1;

/*!
 *  \brief TVM Runtime Device API, abstracts the device
 *  specific interface for memory management.
 */
class DeviceAPI {
public:
    /*! \brief virtual destructor */
    virtual ~DeviceAPI() = default;

    /*!
   * \brief Set the environment device id to device
   * \param dev The device to be set.
   */
    virtual void SetDevice(Device dev) = 0;

    /*!
   * \brief Get attribute of specified device.
   * \param dev The device device
   * \param kind The result kind
   * \param rv The return value.
   * \sa DeviceAttrKind
   */
    //     virtual void GetAttr(Device dev, DeviceAttrKind kind, TVMRetValue* rv) = 0;


    /*!
   * \brief Free a data space on device.
   * \param dev The device device to perform operation.
   * \param ptr The data space.
   */
    virtual void FreeDataSpace(Device dev, void* ptr) = 0;

    /*!
   * \brief Get device API based on device.
   * \param dev The device
   * \param allow_missing Whether allow missing
   * \return The corresponding device API.
   */
    static DeviceAPI* Get(Device dev, bool allow_missing = false);
};

}// namespace litetvm::runtime

#endif//DEVICE_API_H
