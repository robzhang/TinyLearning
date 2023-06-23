//
// Created by Fangbo Zhang on 2023/6/9.
//

#ifndef TINYLEARNING_COS_H
#define TINYLEARNING_COS_H

#include <memory>

#include "../function.h"

namespace TinyLearning {
    class Cos : public Function {
    public:

        vector<shared_ptr<Variable>> Backward(const vector<shared_ptr<Variable>>&) override;

        const char* Name() const override {
            return "Cos";
        }

        static shared_ptr<Cos>New() {
            auto square = new Cos;
            return shared_ptr<Cos>(square);
        }

    private:
        Cos() = default;

        vector<shared_ptr<Tensor>> Forward(const shared_ptr<Tensor>&) override;
    };
}

#endif //TINYLEARNING_COS_H
