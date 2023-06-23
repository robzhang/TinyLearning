//
// Created by Fangbo Zhang on 2023/6/17.
//
#include <vector>

#include "core/layers/linear.h"
#include "core/functions.h"
#include "utils/utils.h"

using namespace std;

namespace TinyLearning {
    Linear::Linear(int inSize, int outSize) {
        this->W_ = Variable::New(vector<int>{inSize, outSize}, ScaledNormalRandomData(inSize * outSize, 0.01/*sqrtf(float(1.0/inSize))*/), "W");
        this->b_ = Variable::Zeros(vector<int>{outSize}, "b");
    }

    shared_ptr<Variable> Linear::Forward(const shared_ptr<Variable>& x) {
        auto y = linear(x, this->W_, this->b_);
        return y;
    }

    vector<shared_ptr<Variable>> Linear::Parameters() {
        return vector<shared_ptr<Variable>>{this->W_, this->b_};
    }
}