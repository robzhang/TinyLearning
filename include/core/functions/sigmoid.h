//
// Created by Fangbo Zhang on 2023/6/15.
//

#ifndef TINYLEARNING_SIGMOID_H
#define TINYLEARNING_SIGMOID_H

#include <memory>

#include "../function.h"

namespace TinyLearning {
    class Sigmoid : public Function {
    public:
        vector<shared_ptr<Variable>> Backward(const vector<shared_ptr<Variable>>&) override;

        const char* Name() const override {
            return "Sigmoid";
        }

        static shared_ptr<Sigmoid>New() {
            auto square = new Sigmoid;
            return shared_ptr<Sigmoid>(square);
        }

    private:
        Sigmoid() = default;

        vector<shared_ptr<Tensor>> Forward(const shared_ptr<Tensor>&) override;
    };
}

#endif //TINYLEARNING_SIGMOID_H
