//
// Created by Fangbo Zhang on 2023/6/11.
//
#include "core/tensor/tensor.h"
#include "core/functions/sum.h"

#include "core/functions.h"

namespace TinyLearning {
    vector<shared_ptr<Tensor>> Sum::Forward(const shared_ptr<Tensor>& x) {
        this->x_shape_ = x->Shape();

        auto y = x->Sum(this->axes_, this->keepDims_);

        return vector<shared_ptr<Tensor>>{shared_ptr<Tensor>(y)};
    }

    vector<shared_ptr<Variable>> Sum::Backward(const vector<shared_ptr<Variable>>& gy) {
        auto gy0 = reshapeBackwardGrad(gy[0]);
        auto gx = broadcastTo(gy0, this->x_shape_);
        return vector<shared_ptr<Variable>>{gx};
    }

    shared_ptr<Variable> Sum::reshapeBackwardGrad(const shared_ptr<Variable>& gy) {
        auto nDim = this->x_shape_.size();

        if (nDim == 0 || this->axes_.empty() || keepDims_) {
            return gy;
        }

        vector<int> shape(this->x_shape_.size(), 0);
        for (auto axis : this->axes_) {
            shape[axis] = 1;
        }

        int off = 0;
        const vector<int>& gyShape = gy->Shape();
        for (int& i : shape) {
            if (i == 0) {
                i = gyShape[off++];
            }
        }

        return reshape(gy, shape);
    }
}