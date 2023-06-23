//
// Created by Fangbo Zhang on 2023/6/10.
//
#include "core/tensor/tensor.h"
#include "core/functions/transpose.h"

#include "core/functions.h"

namespace TinyLearning {
    vector<shared_ptr<Tensor>> Transpose::Forward(const shared_ptr<Tensor>& x) {
        auto y = x->Transpose(this->axes_);

        return vector<shared_ptr<Tensor>>{shared_ptr<Tensor>(y)};
    }

    vector<shared_ptr<Variable>> Transpose::Backward(const vector<shared_ptr<Variable>>& gy) {
        //auto y = this->Output()[0].lock();

        auto it = inverseTransposeAxes(this->axes_);

        return vector<shared_ptr<Variable>>{transpose(gy[0], it)};
    }

    vector<int> Transpose::inverseTransposeAxes(const vector<int>& axes) {
        vector<int> it(axes.size());

        int i = 0;
        for (auto axis : axes) {
            it[axis] = i++;
        }

        return it;
    }
}