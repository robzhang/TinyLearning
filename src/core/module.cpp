//
// Created by Fangbo Zhang on 2023/6/17.
//
#include "core/module.h"

namespace TinyLearning {
    void Module::clearGrads() {
        for (const auto& parameter : this->Parameters()) {
            parameter->ClearGrad();
        }
    }

    shared_ptr<Variable> Module::Forward(const shared_ptr<Variable>&) {
        throw std::runtime_error("Forward(x) NOT implemented");
    }
    shared_ptr<Variable> Module::Forward(const shared_ptr<Variable>&, const shared_ptr<Variable>&) {
        throw std::runtime_error("Forward(x,y) NOT implemented");
    }
    shared_ptr<Variable> Module::Forward(const shared_ptr<Variable>&, const shared_ptr<Variable>&, const shared_ptr<Variable>&) {
        throw std::runtime_error("Forward(x,y,z) NOT implemented");
    }
}