//
// Created by Fangbo Zhang on 2023/6/9.
//
#include "core/tensor/tensor.h"
#include "core/functions/tanh.h"

namespace TinyLearning {
    vector<shared_ptr<Tensor>> Tanh::Forward(const shared_ptr<Tensor>& x) {
        auto y = x->Tanh();

        return vector<shared_ptr<Tensor>>{shared_ptr<Tensor>(y)};
    }

    vector<shared_ptr<Variable>> Tanh::Backward(const vector<shared_ptr<Variable>>& gy) {
        auto y = this->Output()[0].lock();
        auto gx = gy[0] * (1 - y * y);

        return vector<shared_ptr<Variable>>{gx};
    }
}