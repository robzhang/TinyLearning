//
// Created by Fangbo Zhang on 2023/6/15.
//

#ifndef TINYLEARNING_MEANSQUAREDERROR_H
#define TINYLEARNING_MEANSQUAREDERROR_H

#include <memory>

#include "../function.h"

namespace TinyLearning {
    class MeanSquaredError : public Function {
    public:
        vector<shared_ptr<Variable>> Backward(const vector<shared_ptr<Variable>>&) override;

        const char* Name() const override {
            return "Add";
        }

        static shared_ptr<MeanSquaredError>New() {
            auto square = new MeanSquaredError;
            return shared_ptr<MeanSquaredError>(square);
        }

    private:
        MeanSquaredError() = default;

        vector<shared_ptr<Tensor>> Forward(const shared_ptr<Tensor>&,const shared_ptr<Tensor>&) override;
    };
}

#endif //TINYLEARNING_MEANSQUAREDERROR_H
