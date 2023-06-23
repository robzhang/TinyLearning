//
// Created by Fangbo Zhang on 2023/6/9.
//

#include "core/tensor/tensor.h"
#include "core/functions/cos.h"

#include "core/functions.h"

namespace TinyLearning {
    vector<shared_ptr<Tensor>> Cos::Forward(const shared_ptr<Tensor>& x) {
        auto y = x->Cos();

        return vector<shared_ptr<Tensor>>{shared_ptr<Tensor>(y)};
    }

    vector<shared_ptr<Variable>> Cos::Backward(const vector<shared_ptr<Variable>>& gy) {
        auto x = this->Input()[0];
        auto gx = gy[0] * (-sin(x));

        return vector<shared_ptr<Variable>>{gx};
    }
}