//
// Created by Fangbo Zhang on 2023/6/15.
//
#include "core/tensor/tensor.h"
#include "core/functions/linear.h"

#include "core/functions.h"

namespace TinyLearning {
    vector<shared_ptr<Tensor>> LinearFunction::Forward(const shared_ptr<Tensor>& x, const shared_ptr<Tensor>& W, const shared_ptr<Tensor>& b) {
        auto y = Tensor::MatMul(x, W) + b;

        return vector<shared_ptr<Tensor>>{y};
    }

    vector<shared_ptr<Variable>> LinearFunction::Backward(const vector<shared_ptr<Variable>>& gy) {
        auto x = this->Input()[0];
        auto W = this->Input()[1];
        auto b = this->Input()[2];

        auto gx = matMul(gy[0], transpose(W, W->Data()->AxesForTransposingLastTwoDims()));
        auto gW = matMul(transpose(x, x->Data()->AxesForTransposingLastTwoDims()), gy[0]);
        auto gb = sumTo(gy[0], b->Shape());

        return vector<shared_ptr<Variable>>{gx, gW, gb};
    }
}