//
// Created by Fangbo Zhang on 2023/6/10.
//

#ifndef TINYLEARNING_RESHAPE_H
#define TINYLEARNING_RESHAPE_H

#include <memory>
#include <utility>

#include "../function.h"

namespace TinyLearning {
    class Reshape : public Function {
    public:
        vector<shared_ptr<Variable>> Backward(const vector<shared_ptr<Variable>>&) override;

        const char* Name() const override {
            return "Reshape";
        }

        static shared_ptr<Reshape>New(vector<int> y_shape) {
            auto reshape = new Reshape(std::move(y_shape));
            return shared_ptr<Reshape>(reshape);
        }

    private:
        explicit Reshape(vector<int> y_shape) : y_shape_(std::move(y_shape)){}

        vector<int> x_shape_;
        vector<int> y_shape_;

        vector<shared_ptr<Tensor>> Forward(const shared_ptr<Tensor>&) override;
    };
}

#endif //TINYLEARNING_RESHAPE_H
