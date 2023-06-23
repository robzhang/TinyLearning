//
// Created by Fangbo Zhang on 2023/6/7.
//
#include "core/tensor/tensor.h"
#include "core/functions/pow.h"

namespace TinyLearning {
    vector<shared_ptr<Tensor>> Pow::Forward(const shared_ptr<Tensor>& x) {
        auto y = x ^ C_;

        return vector<shared_ptr<Tensor>>{y};
    }

    vector<shared_ptr<Variable>> Pow::Backward(const vector<shared_ptr<Variable>>& gy) {
        auto x = this->Input()[0];

        auto gx = C_ * (x ^ (C_ - 1)) * gy[0];

        return vector<shared_ptr<Variable>>{gx};
    }
}