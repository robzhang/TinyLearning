//
// Created by Fangbo Zhang on 2023/6/10.
//
#include "core/tensor/tensor.h"
#include "core/functions/reshape.h"

#include "core/functions.h"

namespace TinyLearning {
    vector<shared_ptr<Tensor>> Reshape::Forward(const shared_ptr<Tensor>& x) {
        this->x_shape_ = x->Shape();

        auto y = x->Reshape(this->y_shape_);

        return vector<shared_ptr<Tensor>>{shared_ptr<Tensor>(y)};
    }

    vector<shared_ptr<Variable>> Reshape::Backward(const vector<shared_ptr<Variable>>& gy) {
        return vector<shared_ptr<Variable>>{reshape(gy[0], this->x_shape_)};
    }
}