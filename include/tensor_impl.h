//
// Created by richard on 6/22/25.
//

#ifndef TENSOR_IMPL_H
#define TENSOR_IMPL_H

#include "cpu_allocator.h"
#include "data_type.h"
#include "storage.h"
#include "tensor_utils.h"

#include <fmt/format.h>
#include <glog/logging.h>
#include <memory>
#include <vector>

namespace atp {

struct TensorInfo {
    void* data{nullptr};
    int32_t ndim{0};
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    DLDataType dtype;
    DeviceType device_type;
};

inline int64_t GetTensorSize(const TensorInfo& t) {
    if (t.shape.empty()) {
        return 0;
    }

    int64_t numel = 1;
    for (int i = 0; i < t.ndim; ++i) {
        numel *= t.shape[i];
    }

    return (numel * t.dtype.bits * t.dtype.lanes + 7) / 8;
}

inline bool _compute_contiguous(const std::vector<int64_t>& shape, const std::vector<int64_t>& strides) {
    if (strides.empty()) {
        return true;
    }

    if (shape.size() != strides.size()) {
        return false;
    }

    int64_t expected_stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        if (shape[i] == 1) {
            continue;
        }

        if (strides[i] != expected_stride) {
            return false;
        }

        expected_stride *= shape[i];
    }
    return true;
}

class Scalar {
public:
    Scalar() : Scalar(static_cast<int64_t>(0)) {}

    template<typename T,
             std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>>* = nullptr>
    Scalar(T val) {
        v.i = static_cast<decltype(v.i)>(val);
        dtype = DLDataTypeCode::kInt;
    }

    template<typename T,
             std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
    Scalar(T val) {
        v.d = static_cast<decltype(v.d)>(val);
        dtype = DLDataTypeCode::kFloat;
    }

    template<typename T,
             std::enable_if_t<std::is_same_v<T, bool>>* = nullptr>
    Scalar(T val) {
        v.i = static_cast<decltype(v.i)>(val);
        dtype = DLDataTypeCode::kBool;
    }

    Scalar(const Scalar& other) : v(other.v), dtype(other.dtype) {}

    Scalar(Scalar&& other) noexcept : v(other.v), dtype(other.dtype) {
        other.v.i = 0;
        other.dtype = DLDataTypeCode::kInt;
    }

    Scalar& operator=(const Scalar& other) {
        Scalar tmp(other);
        swap(*this, tmp);
        return *this;
    }

    Scalar& operator=(Scalar&& other) noexcept {
        Scalar tmp(std::move(other));
        swap(*this, tmp);
        return *this;
    }

    NODISCARD bool isIntegral() const {
        return dtype == DLDataTypeCode::kInt;
    }

    NODISCARD bool isFloatingPoint() const {
        return dtype == DLDataTypeCode::kFloat;
    }

    NODISCARD bool isBool() const {
        return dtype == DLDataTypeCode::kBool;
    }

    NODISCARD DLDataTypeCode type() const {
        return dtype;
    }

    friend void swap(Scalar& a, Scalar& b) noexcept {
        std::swap(a.v, b.v);
        std::swap(a.dtype, b.dtype);
    }

#define ACCESSOR(type, name)                                                 \
    type to##name() const {                                                  \
        if (dtype == DLDataTypeCode::kInt || dtype == DLDataTypeCode::kBool) \
            return static_cast<type>(v.i);                                   \
        else if (dtype == DLDataTypeCode::kFloat)                            \
            return static_cast<type>(v.d);                                   \
        else                                                                 \
            throw std::runtime_error("Unsupported data type");               \
    }
    SCALAR_TYPES_NAME(ACCESSOR);
#undef ACCESSOR

    template<typename T>
    T to() const = delete;

private:
    union val {
        int64_t i;
        double d{};
    } v;

    DLDataTypeCode dtype;
};

#define DEFINE_TO(T, name)           \
    template<>                       \
    inline T Scalar::to<T>() const { \
        return to##name();           \
    }
SCALAR_TYPES_NAME(DEFINE_TO);
#undef DEFINE_TO

/**
 * The low-level representation of a tensor, which contains a pointer to a
 * storage (which contains the actual data) and metadata (e.g., shape and
 * strides) describing this particular view of the data as a tensor.
 *
 * Some basic characteristics about the in-memory representation of tensors:
 *
 * - It contains a pointer to a storage struct (Storage/StorageImpl) which
 *   contains the pointer to the actual data and records the data type and
 *   device of the view. This allows multiple tensors to alias the same
 *   underlying data, which allows efficiently implementing differing *views*
 *   on a tensor.
 *
 * - The tensor struct itself records view-specific metadata about the tensor,
 *   e.g., shape, strides and offset into storage. Each view of a storage can
 *   have a different shape or offset.
 *
 *
 **/
class TensorImpl {
public:
    TensorImpl() = delete;
    TensorImpl(const TensorImpl&) = delete;
    TensorImpl(TensorImpl&&) noexcept = delete;
    TensorImpl& operator=(const TensorImpl&) = delete;
    TensorImpl& operator=(TensorImpl&&) noexcept = delete;

    TensorImpl(const std::vector<int64_t>& shape, int64_t storage_offset, DataType dtype, DeviceType device);

    /**
     * The number of elements in a tensor.
     **/
    NODISCARD int64_t numel() const {
        return numel_;
    }

    NODISCARD bool empty() const {
        return numel_ == 0;
    }

    /**
     * Return the number of dimensions of this tensor.  Note that 0-dimension
     * represents a Tensor that is a Scalar, e.g., one that has a single element.
     **/
    NODISCARD int64_t ndim() const {
        return shape_and_stride_.size();
    }

    /**
     * Return a reference to the shape of this tensor. This reference remains
     * valid as long as the tensor is live and not resized.
     **/
    NODISCARD std::vector<int64_t> shape() const {
        return shape_and_stride_.get_shape();
    }

    /**
     * Return a reference to the strides of this tensor.  This reference remains
     * valid as long as the tensor is live and not restrided.
     **/
    NODISCARD std::vector<int64_t> strides() const {
        return shape_and_stride_.get_strides();
    }

    NODISCARD size_t itemsize() const {
        CHECK(dtype_initialized()) << "Can't get item sizer of Tensor that doesn't have initialized dtype.";
        return dtype_.nbytes();
    }

    NODISCARD bool has_storage() const {
        return storage_;
    }

    /**
     * True if a tensor is storage initialized.  A tensor may become
     * storage UNINITIALIZED after a Resize() or FreeMemory()
     **/
    NODISCARD bool storage_initialized() const {
        CHECK(has_storage()) << "Can't call storage_initialized() on a tensor that doesn't have storage.";
        return storage_.const_data() != nullptr || numel_ == 0;
    }

    NODISCARD bool dtype_initialized() const {
        return dtype_ != DataType();
    }

    NODISCARD const Storage& storage() const {
        return storage_;
    }

    /**
     * Return the offset in number of elements into the storage that this
     * tensor points to.  Most tensors have storage_offset() == 0, but,
     * for example, an index into a tensor will have a non-zero storage_offset().
     *
     * WARNING: This is NOT computed in bytes.
     **/
    NODISCARD int64_t storage_offset() const {
        return storage_offset_;
    }

    NODISCARD DeviceType device() const {
        return storage_.device();
    }

    NODISCARD DataType dtype() const {
        return dtype_;
    }

    NODISCARD bool compute_contiguous() const {
        return _compute_contiguous(shape(), strides());
    }

    /**
   * Whether a tensor is laid out in contiguous memory.
   *
   * Tensors with non-trivial strides are not contiguous. See
   * compute_contiguous() for the exact definition of whether
   * a tensor is contiguous or not.
   */
    NODISCARD bool is_contiguous() const {
        return is_contiguous_;
    }

    /**
   * Return a void* data pointer to the actual data which this tensor refers to.
   *
   * It is invalid to call data() on a dtype-uninitialized tensor, even if the size is 0.
   *
   * WARNING: The data pointed to by this tensor may not contiguous; do NOT
   * assume that itemsize() * numel() is sufficient to compute the bytes that
   * can be validly read from this tensor.
   */
    NODISCARD void* data() const {
        return data_impl<void>(
                [this] {
                    return static_cast<char*>(storage_.data());
                });
    }

    NODISCARD const void* const_data() const {
        return data_impl<const void>(
                [this] {
                    return static_cast<const char*>(storage_.const_data());
                });
    }

    template<typename T>
    T* data_ptr_impl() const {
        auto get_data = [this] {
            return static_cast<T*>(storage_.data());
        };
        return data_ptr_impl_impl<T>(get_data);
    }

    template<typename T>
    const T* const_data_ptr_impl() const {
        auto get_data = [this] {
            return static_cast<const T*>(storage_.const_data());
        };
        return data_ptr_impl_impl<const T>(get_data);
    }

private:
    template<typename Void, typename Func>
    NODISCARD Void* data_impl(const Func& get_data) const {
        CHECK(has_storage()) << "Can't access data pointer of Tensor that doesn't have storage.";
        CHECK(dtype_initialized()) << "Can't access data pointer of Tensor that doesn't have initialized dtype.";
        auto* data = get_data();
        static_assert(sizeof(*data) == 1, "get_data must return a byte-addressed pointer.");
        if (empty()) {
            return nullptr;
        }

        return data + dtype_.nbytes() * storage_offset_;
    }

    // Shared implementation of data_ptr_impl() and the const_data_ptr_impl().
    template<typename T, typename Func>
    __ubsan_ignore_pointer_overflow__ T* data_ptr_impl_impl(const Func& get_data) const {
        CHECK(has_storage()) << "Can't access data pointer of Tensor that doesn't have storage.";
        CHECK(storage_initialized() && dtype_.Match<std::remove_cv_t<T>>())
                << "The tensor has a non-zero number of elements, but its data is not allocated yet.";
        return get_data() + storage_offset_;
    }

    void init_bitfield() {
        is_contiguous_ = true;
    }

    Storage storage_;
    // The offset in number of elements into the storage that this tensor points to.
    int64_t storage_offset_ = 0;

    // If shape and strides are empty, the numel is 1!! However, most of the
    // time, we will immediately set the shape to {0} and reset numel to 0.
    // (Can't do that in the default initializers, because there's no way to
    // spell "allocate a one-element array" for strides_).
    int64_t numel_ = 1;
    DataType dtype_;
    ShapeAndStride shape_and_stride_;

    bool is_contiguous_ : 1;

    // Tensor is stored in the channels last 2d memory format, when dimensions
    // order is (N)CHW and C-strides < W-strides < H-strides (< N-strides)
    // (If size of any dimension is equal to 1, this dimension strides value
    // is not taken into account).
    // bool is_channels_last_ : 1;
};

}// namespace atp

#endif//TENSOR_IMPL_H
