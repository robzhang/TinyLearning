//
// Created by Fangbo Zhang on 2023/6/5.
//

#include "core/tensor/tensor.h"
#include "core/functions/exp.h"

#include "core/functions.h"

namespace TinyLearning {
    vector<shared_ptr<Tensor>> Exp::Forward(const shared_ptr<Tensor>& x) {
        auto y = x->Exp();

        return vector<shared_ptr<Tensor>>{shared_ptr<Tensor>(y)};
    }

    vector<shared_ptr<Variable>> Exp::Backward(const vector<shared_ptr<Variable>>& gy) {
        auto x = this->Input()[0];
        auto gx = gy[0] * exp(x);

        return vector<shared_ptr<Variable>>{gx};
    }
}