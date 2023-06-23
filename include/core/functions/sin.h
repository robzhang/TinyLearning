//
// Created by Fangbo Zhang on 2023/6/9.
//

#ifndef TINYLEARNING_SIN_H
#define TINYLEARNING_SIN_H

#include <memory>

#include "../function.h"

namespace TinyLearning {
    class Sin : public Function {
    public:
        vector<shared_ptr<Variable>> Backward(const vector<shared_ptr<Variable>>&) override;

        const char* Name() const override {
            return "Sin";
        }

        static shared_ptr<Sin>New() {
            auto square = new Sin;
            return shared_ptr<Sin>(square);
        }

    private:
        Sin() = default;

        vector<shared_ptr<Tensor>> Forward(const shared_ptr<Tensor>&) override;
    };
}

#endif //TINYLEARNING_SIN_H
