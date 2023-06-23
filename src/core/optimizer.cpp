//
// Created by Fangbo Zhang on 2023/6/19.
//

#include "core/optimizer.h"

namespace TinyLearning {
    Optimizer::Optimizer(const vector<shared_ptr<Variable>>& parameters, float learningRate) : parameters_(parameters), learningRate_(learningRate) {

    }

    void Optimizer::Update() {
        for (auto& parameter : this->parameters_) {
            updateOne(parameter);
        }
    }

    void Optimizer::ClearGrads() {
        for (const auto& parameter : this->parameters_) {
            parameter->ClearGrad();
        }
    }
}