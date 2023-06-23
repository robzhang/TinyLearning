//
// Created by Fangbo Zhang on 2023/6/6.
//

#ifndef TINYLEARNING_ADD_H
#define TINYLEARNING_ADD_H

#include <memory>

#include "../function.h"

namespace TinyLearning {
    class Add : public Function {
    public:

        vector<shared_ptr<Variable>> Backward(const vector<shared_ptr<Variable>>&) override;

        const char* Name() const override {
            return "Add";
        }

        static shared_ptr<Add>New() {
            auto square = new Add;
            return shared_ptr<Add>(square);
        }

    private:
        Add() = default;
        vector<shared_ptr<Tensor>> Forward(const shared_ptr<Tensor>&,const shared_ptr<Tensor>&) override;
    };
}

#endif //TINYLEARNING_ADD_H
