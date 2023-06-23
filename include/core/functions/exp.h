//
// Created by Fangbo Zhang on 2023/6/5.
//

#ifndef TINYLEARNING_EXP_H
#define TINYLEARNING_EXP_H

#include "../function.h"

namespace TinyLearning {
    class Exp : public Function {
    public:

        vector<shared_ptr<Variable>> Backward(const vector<shared_ptr<Variable>>&) override;

        const char* Name() const override {
            return "Exp";
        }

        static shared_ptr<Exp>New() {
            auto exp = new Exp;
            return shared_ptr<Exp>(exp);
        }

    private:
        Exp() = default;

        vector<shared_ptr<Tensor>> Forward(const shared_ptr<Tensor>&) override;
    };
}

#endif //TINYLEARNING_EXP_H
