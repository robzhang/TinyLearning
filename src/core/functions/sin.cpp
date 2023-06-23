//
// Created by Fangbo Zhang on 2023/6/9.
//
#include "core/tensor/tensor.h"
#include "core/functions/sin.h"

#include "core/functions.h"

namespace TinyLearning {
    vector<shared_ptr<Tensor>> Sin::Forward(const shared_ptr<Tensor>& x) {
        auto y = x->Sin();

        return vector<shared_ptr<Tensor>>{shared_ptr<Tensor>(y)};
    }

    vector<shared_ptr<Variable>> Sin::Backward(const vector<shared_ptr<Variable>>& gy) {
        auto x = this->Input()[0];
        auto gx = gy[0] * cos(x);

        return vector<shared_ptr<Variable>>{gx};
    }
}