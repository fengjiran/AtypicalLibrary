//
// Created by richard on 1/31/25.
//

#include "runtime/ndarray.h"

namespace litetvm::runtime {
ShapeTuple NDArray::Shape() const {
    return static_cast<const Container*>(data_.get())->shape_;
}

DataType NDArray::DataType() const {
    return runtime::DataType(get_mutable()->dl_tensor.dtype);
}

}