//
// Created by richard on 6/28/25.
//
#include "storage.h"

namespace atp {

template<size_t MAX_INLINE_SIZE>
void ShapeAndStride<MAX_INLINE_SIZE>::resize_slow_path(const size_t new_size, const size_t old_size) {
    if (new_size <= MAX_INLINE_SIZE) {
        CHECK(!is_inline()) << "resize slow path called when fast path should have been hit!";
        auto* tmp = outline_storage_;
        memcpy(&inline_storage_[0], &tmp[0], MAX_INLINE_SIZE * sizeof(inline_storage_[0]));
        memcpy(&inline_storage_[MAX_INLINE_SIZE], &tmp[old_size], MAX_INLINE_SIZE * sizeof(inline_storage_[0]));
        free(tmp);
    } else {
        if (is_inline()) {
            auto* tmp = static_cast<int64_t*>(malloc(storage_bytes(new_size)));
            CHECK(tmp) << "Could not allocate memory for Tensor ShapeAndStride.";
            const auto bytes_to_copy = old_size * sizeof(inline_storage_[0]);
            const auto bytes_to_zero = new_size > old_size ? (new_size - old_size) * sizeof(tmp[0]) : 0;
            memcpy(&tmp[0], &inline_storage_[0], bytes_to_copy);
            if (bytes_to_zero) {
                memset(&tmp[old_size], 0, bytes_to_zero);
            }

            memcpy(&tmp[new_size], &inline_storage_[MAX_INLINE_SIZE], bytes_to_copy);
            if (bytes_to_zero) {
                memset(&tmp[new_size + old_size], 0, bytes_to_zero);
            }

            outline_storage_ = tmp;
        } else {
            const bool is_growing = new_size > old_size;
            if (is_growing) {
                resize_outline_storage(new_size);
            }

            memmove(outline_storage_ + new_size, outline_storage_ + old_size,
                    std::min(new_size, old_size) * sizeof(outline_storage_[0]));

            if (is_growing) {
                const auto bytes_to_zero = (new_size - old_size) * sizeof(outline_storage_[0]);
                memset(&outline_storage_[old_size], 0, bytes_to_zero);
                memset(&outline_storage_[new_size + old_size], 0, bytes_to_zero);
            } else {
                resize_outline_storage(new_size);
            }
        }
    }
    size_ = new_size;
}


}// namespace atp