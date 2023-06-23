//
// Created by Fangbo Zhang on 2023/6/7.
//

#ifndef TINYLEARNING_DIV_H
#define TINYLEARNING_DIV_H

#include <memory>

#include "../function.h"

namespace TinyLearning {
    class Div : public Function {
    public:

        vector<shared_ptr<Variable>> Backward(const vector<shared_ptr<Variable>>&) override;

        const char* Name() const override {
            return "Div";
        }

        static shared_ptr<Div>New() {
            auto div = new Div;
            return shared_ptr<Div>(div);
        }

    private:
        Div() = default;

        vector<shared_ptr<Tensor>> Forward(const shared_ptr<Tensor>&,const shared_ptr<Tensor>&) override;
    };
}

#endif //TINYLEARNING_DIV_H
