//
// Created by Fangbo Zhang on 2023/6/15.
//
#include "core/tensor/tensor.h"
#include "core/functions/matMul.h"

#include "core/functions.h"

namespace TinyLearning {
    vector<shared_ptr<Tensor>> MatMul::Forward(const shared_ptr<Tensor>& x, const shared_ptr<Tensor>& W) {
        auto y = Tensor::MatMul(x, W);

        return vector<shared_ptr<Tensor>>{y};
    }

    vector<shared_ptr<Variable>> MatMul::Backward(const vector<shared_ptr<Variable>>& gy) {
        auto x = this->Input()[0];
        auto W = this->Input()[1];

        auto gx = matMul(gy[0], transpose(W, W->Data()->AxesForTransposingLastTwoDims()));
        auto gW = matMul(transpose(x, x->Data()->AxesForTransposingLastTwoDims()), gy[0]);

        return vector<shared_ptr<Variable>>{gx, gW};
    }
}