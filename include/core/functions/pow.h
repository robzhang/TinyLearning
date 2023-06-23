//
// Created by Fangbo Zhang on 2023/6/7.
//

#ifndef TINYLEARNING_POW_H
#define TINYLEARNING_POW_H

#include <memory>

#include "../function.h"

namespace TinyLearning {
    class Pow : public Function {
    public:
        vector<shared_ptr<Variable>> Backward(const vector<shared_ptr<Variable>>&) override;

        const char* Name() const override {
            return "Pow";
        }

        static shared_ptr<Pow>New(float c) {
            auto pow = new Pow(c);
            return shared_ptr<Pow>(pow);
        }

    private:
        explicit Pow(float c) : C_(c){}

        float C_;

        vector<shared_ptr<Tensor>> Forward(const shared_ptr<Tensor>&) override;
    };
}

#endif //TINYLEARNING_POW_H
