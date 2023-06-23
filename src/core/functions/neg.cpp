//
// Created by Fangbo Zhang on 2023/6/7.
//
#include "core/tensor/tensor.h"
#include "core/functions/neg.h"

namespace TinyLearning {
    vector<shared_ptr<Tensor>> Neg::Forward(const shared_ptr<Tensor>& x) {
        auto y = -1 * x;

        return vector<shared_ptr<Tensor>>{y};
    }

    vector<shared_ptr<Variable>> Neg::Backward(const vector<shared_ptr<Variable>>& gy) {
        auto dx = -1 * gy[0];

        return vector<shared_ptr<Variable>>{dx};
    }
}