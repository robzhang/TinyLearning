//
// Created by Fangbo Zhang on 2023/6/15.
//
#include "core/tensor/tensor.h"
#include "core/functions/sigmoid.h"

namespace TinyLearning {
    vector<shared_ptr<Tensor>> Sigmoid::Forward(const shared_ptr<Tensor>& x) {
        auto y = (*((*(x * 0.5)->Tanh()) * 0.5)) + 0.5;

        return vector<shared_ptr<Tensor>>{shared_ptr<Tensor>(y)};
    }

    vector<shared_ptr<Variable>> Sigmoid::Backward(const vector<shared_ptr<Variable>>& gy) {
        auto y = this->Output()[0].lock();
        auto gx = gy[0] * y * (1 - y);

        return vector<shared_ptr<Variable>>{gx};
    }
}