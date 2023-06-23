//
// Created by Fangbo Zhang on 2023/6/7.
//
#include "core/tensor/tensor.h"
#include "core/functions/div.h"

#include "core/functions.h"

namespace TinyLearning {
    vector<shared_ptr<Tensor>> Div::Forward(const shared_ptr<Tensor>& x0, const shared_ptr<Tensor>& x1) {
        auto y = x0 / x1;

        return vector<shared_ptr<Tensor>>{y};
    }

    vector<shared_ptr<Variable>> Div::Backward(const vector<shared_ptr<Variable>>& gy) {
        auto x0 = this->Input()[0];
        auto x1 = this->Input()[1];
        auto gx0 = gy[0] / x1;
        auto gx1 = gy[0] * ((-1 * x0) / (x1 ^ 2));

        if (gx0->Shape() != x0->Shape()) {
            gx0 = sumTo(gx0, x0->Shape());
        }
        if (gx1->Shape() != x1->Shape()) {
            gx1 = sumTo(gx1, x1->Shape());
        }

        return vector<shared_ptr<Variable>>{gx0, gx1};
    }
}