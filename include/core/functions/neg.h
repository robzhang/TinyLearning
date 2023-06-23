//
// Created by Fangbo Zhang on 2023/6/7.
//

#ifndef TINYLEARNING_NEG_H
#define TINYLEARNING_NEG_H

#include <memory>

#include "../function.h"

namespace TinyLearning {
    class Neg : public Function {
    public:
        vector<shared_ptr<Variable>> Backward(const vector<shared_ptr<Variable>>&) override;

        const char* Name() const override {
            return "Neg";
        }

        static shared_ptr<Neg>New() {
            auto neg = new Neg;
            return shared_ptr<Neg>(neg);
        }

    private:
        Neg() = default;

        vector<shared_ptr<Tensor>> Forward(const shared_ptr<Tensor>&) override;
    };
}

#endif //TINYLEARNING_NEG_H
