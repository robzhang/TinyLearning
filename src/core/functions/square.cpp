//
// Created by Fangbo Zhang on 2023/6/5.
//

#include "core/tensor/tensor.h"
#include "core/functions/square.h"

namespace TinyLearning {
    vector<shared_ptr<Tensor>> Square::Forward(const shared_ptr<Tensor>& x) {
        auto y = x ^ 2;

        return vector<shared_ptr<Tensor>>{y};
    }

    vector<shared_ptr<Variable>> Square::Backward(const vector<shared_ptr<Variable>>& gy) {
        auto dx = 2 * this->Input()[0] * gy[0];

        return vector<shared_ptr<Variable>>{dx};
    }
}