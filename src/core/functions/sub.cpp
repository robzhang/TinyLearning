//
// Created by Fangbo Zhang on 2023/6/7.
//
#include "core/tensor/tensor.h"
#include "core/functions/sub.h"

#include "core/functions.h"

namespace TinyLearning {
    vector<shared_ptr<Tensor>> Sub::Forward(const shared_ptr<Tensor>& x0, const shared_ptr<Tensor>& x1) {
        auto y = x0 - x1;

        return vector<shared_ptr<Tensor>>{y};
    }

    vector<shared_ptr<Variable>> Sub::Backward(const vector<shared_ptr<Variable>>& gy) {
        auto gx0 = gy[0];
        auto gx1 = -gy[0];

        if (gx0->Shape() != this->Input()[0]->Shape()) {
            gx0 = sumTo(gx0, this->Input()[0]->Shape());
        }
        if (gx1->Shape() != this->Input()[1]->Shape()) {
            gx1 = sumTo(gx1, this->Input()[1]->Shape());
        }

        return vector<shared_ptr<Variable>>{gx0, gx1};
    }
}