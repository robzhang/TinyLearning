//
// Created by Fangbo Zhang on 2023/6/5.
//

#ifndef TINYLEARNING_NUMERICAL_DIFF_H
#define TINYLEARNING_NUMERICAL_DIFF_H

#include <functional>

#include "variable.h"
#include "function.h"

namespace TinyLearning {
    Tensor* numerical_diff(const std::function<Variable(Variable&)>& f, Variable& x, float eps = 1e-4);
}

#endif //TINYLEARNING_NUMERICAL_DIFF_H
