//
// Created by Fangbo Zhang on 2023/6/19.
//
#include "core/optimizers/sgd.h"

namespace TinyLearning {
    SGD::SGD(const vector<shared_ptr<Variable>>& parameters, float learningRate) : Optimizer(parameters, learningRate) {

    }

    void SGD::updateOne(shared_ptr<Variable>& parameter) {
        parameter->Data() -= this->LearningRate() * parameter->Grad()->Data();
    }
}