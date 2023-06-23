//
// Created by Fangbo Zhang on 2023/6/9.
//

#ifndef TINYLEARNING_TANH_H
#define TINYLEARNING_TANH_H

#include <memory>

#include "../function.h"

namespace TinyLearning {
    class Tanh : public Function {
    public:
        vector<shared_ptr<Variable>> Backward(const vector<shared_ptr<Variable>>&) override;

        const char* Name() const override {
            return "Tanh";
        }

        static shared_ptr<Tanh>New() {
            auto square = new Tanh;
            return shared_ptr<Tanh>(square);
        }

    private:
        Tanh() = default;

        vector<shared_ptr<Tensor>> Forward(const shared_ptr<Tensor>&) override;
    };
}

#endif //TINYLEARNING_TANH_H
