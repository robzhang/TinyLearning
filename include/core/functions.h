//
// Created by Fangbo Zhang on 2023/6/5.
//

#ifndef TINYLEARNING_FUNCTIONS_H
#define TINYLEARNING_FUNCTIONS_H

namespace TinyLearning {
    shared_ptr<Variable> square(const shared_ptr<Variable>& x);
    shared_ptr<Variable> exp(const shared_ptr<Variable>& x);
    shared_ptr<Variable> add(const shared_ptr<Variable>& x0, const shared_ptr<Variable>& x1);
    shared_ptr<Variable> mul(const shared_ptr<Variable>& x0, const shared_ptr<Variable>& x1);
    shared_ptr<Variable> neg(const shared_ptr<Variable>& x);
    shared_ptr<Variable> sub(const shared_ptr<Variable>& x0, const shared_ptr<Variable>& x1);
    shared_ptr<Variable> div(const shared_ptr<Variable>& x0, const shared_ptr<Variable>& x1);
    shared_ptr<Variable> pow(const shared_ptr<Variable>& x, float c);
    shared_ptr<Variable> sin(const shared_ptr<Variable>& x);
    shared_ptr<Variable> cos(const shared_ptr<Variable>& x);
    shared_ptr<Variable> tanh(const shared_ptr<Variable>& x);
    shared_ptr<Variable> reshape(const shared_ptr<Variable>& x, const vector<int>& shape);
    shared_ptr<Variable> transpose(const shared_ptr<Variable>& x, const vector<int>& axes);
    shared_ptr<Variable> sum(const shared_ptr<Variable>& x, const vector<int>& axes = vector<int>{}, bool keepDims = false);
    shared_ptr<Variable> broadcastTo(const shared_ptr<Variable>& x, const vector<int>& shape);
    shared_ptr<Variable> sumTo(const shared_ptr<Variable>& x, const vector<int>& shape);
    shared_ptr<Variable> matMul(const shared_ptr<Variable>& x, const shared_ptr<Variable>& W);
    shared_ptr<Variable> meanSquaredError(const shared_ptr<Variable>& x0, const shared_ptr<Variable>& x1);
    shared_ptr<Variable> linear(const shared_ptr<Variable>& x, const shared_ptr<Variable>& W, const shared_ptr<Variable>& b);
    shared_ptr<Variable> sigmoid(const shared_ptr<Variable>& x);
}

#endif //TINYLEARNING_FUNCTIONS_H
