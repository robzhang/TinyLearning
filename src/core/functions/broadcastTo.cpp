//
// Created by Fangbo Zhang on 2023/6/12.
//
#include "core/tensor/tensor.h"
#include "core/functions/broadcastTo.h"

#include "core/functions.h"

namespace TinyLearning {
    vector<shared_ptr<Tensor>> BroadcastTo::Forward(const shared_ptr<Tensor>& x) {
        this->x_shape_ = x->Shape();

        auto y = x->BroadcastTo(this->y_shape_);

        return vector<shared_ptr<Tensor>>{shared_ptr<Tensor>(y)};
    }

    vector<shared_ptr<Variable>> BroadcastTo::Backward(const vector<shared_ptr<Variable>>& gy) {
        auto gx = sumTo(gy[0], this->x_shape_);
        return vector<shared_ptr<Variable>>{gx};
    }
}