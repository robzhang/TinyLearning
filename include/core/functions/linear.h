//
// Created by Fangbo Zhang on 2023/6/15.
//

#ifndef TINYLEARNING_LINEAR_H
#define TINYLEARNING_LINEAR_H

#include <memory>

#include "../function.h"

namespace TinyLearning {
    class LinearFunction : public Function {
    public:
        vector<shared_ptr<Variable>> Backward(const vector<shared_ptr<Variable>>&) override;

        const char* Name() const override {
            return "LinearFunction";
        }

        static shared_ptr<LinearFunction>New() {
            auto square = new LinearFunction;
            return shared_ptr<LinearFunction>(square);
        }

    private:
        LinearFunction() = default;

        vector<shared_ptr<Tensor>> Forward(const shared_ptr<Tensor>&, const shared_ptr<Tensor>&, const shared_ptr<Tensor>&) override;

    };
}

#endif //TINYLEARNING_LINEAR_H
