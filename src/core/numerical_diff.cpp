//
// Created by Fangbo Zhang on 2023/6/5.
//
#include <memory>

#include "core/numerical_diff.h"

using namespace std;

namespace TinyLearning {
    Tensor* numerical_diff(const function<Variable(Variable&)>& f, Variable& x, float eps) {
        auto x0 = Variable(shared_ptr<Tensor>(*(x.Data()) - eps), "");
        auto x1 = Variable(shared_ptr<Tensor>(*(x.Data()) + eps), "");

        auto diff = *(*f(x1).Data() - *f(x0).Data()) / (2 * eps);

        return diff;
    }
}