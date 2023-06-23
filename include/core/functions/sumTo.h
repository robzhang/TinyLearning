//
// Created by Fangbo Zhang on 2023/6/12.
//

#ifndef TINYLEARNING_SUMTO_H
#define TINYLEARNING_SUMTO_H

#include <memory>
#include <utility>

#include "../function.h"

namespace TinyLearning {
    class SumTo : public Function {
    public:
        vector<shared_ptr<Variable>> Backward(const vector<shared_ptr<Variable>>&) override;

        const char* Name() const override {
            return "SumTo";
        }

        static shared_ptr<SumTo>New(vector<int> y_shape) {
            auto sumTo = new SumTo(std::move(y_shape));
            return shared_ptr<SumTo>(sumTo);
        }

    private:
        explicit SumTo(vector<int> y_shape) : y_shape_(std::move(y_shape)){}

        vector<int> x_shape_;
        vector<int> y_shape_;

        vector<shared_ptr<Tensor>> Forward(const shared_ptr<Tensor>&) override;
    };
}

#endif //TINYLEARNING_SUMTO_H
