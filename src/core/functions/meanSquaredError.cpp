//
// Created by Fangbo Zhang on 2023/6/15.
//
#include "core/tensor/tensor.h"
#include "core/functions/meanSquaredError.h"

namespace TinyLearning {
    vector<shared_ptr<Tensor>> MeanSquaredError::Forward(const shared_ptr<Tensor>& x0, const shared_ptr<Tensor>& x1) {
        auto diff = x0 - x1;
        auto y = *(diff ^ 2)->Sum() / float(diff->Size());

        return vector<shared_ptr<Tensor>>{shared_ptr<Tensor>(y)};
    }

    vector<shared_ptr<Variable>> MeanSquaredError::Backward(const vector<shared_ptr<Variable>>& gy) {
        auto x0 = this->Input()[0];
        auto x1 = this->Input()[1];

        auto diff = x0 - x1;

        auto gx0 = gy[0] * diff * (2.0f / float(diff->Size()));
        auto gx1 = -gx0;

        return vector<shared_ptr<Variable>>{gx0, gx1};
    }
}