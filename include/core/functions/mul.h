//
// Created by Fangbo Zhang on 2023/6/7.
//

#ifndef TINYLEARNING_MUL_H
#define TINYLEARNING_MUL_H

#include <memory>

#include "../function.h"

namespace TinyLearning {
    class Mul : public Function {
    public:
        vector<shared_ptr<Variable>> Backward(const vector<shared_ptr<Variable>>&) override;

        const char* Name() const override {
            return "Mul";
        }

        static shared_ptr<Mul>New() {
            auto mul = new Mul;
            return shared_ptr<Mul>(mul);
        }

    private:
        Mul() = default;

        vector<shared_ptr<Tensor>> Forward(const shared_ptr<Tensor>&, const shared_ptr<Tensor>&) override;
    };
}

#endif //TINYLEARNING_MUL_H
