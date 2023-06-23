//
// Created by Fangbo Zhang on 2023/6/11.
//

#ifndef TINYLEARNING_SUM_H
#define TINYLEARNING_SUM_H

#include <memory>
#include <utility>

#include "../function.h"

namespace TinyLearning {
    class Sum : public Function {
    public:
        vector<shared_ptr<Variable>> Backward(const vector<shared_ptr<Variable>>&) override;

        const char* Name() const override {
            return "Sum";
        }

        static shared_ptr<Sum>New(const vector<int>& axis = vector<int>{}, bool keepDims = false) {
            auto sum = new Sum(axis, keepDims);
            return shared_ptr<Sum>(sum);
        }

    private:
        explicit Sum(const vector<int>& axis, bool keepDims) : axes_(axis), keepDims_(keepDims){}

        shared_ptr<Variable> reshapeBackwardGrad(const shared_ptr<Variable>&);

        vector<int> axes_;
        bool keepDims_;

        vector<int> x_shape_;

        vector<shared_ptr<Tensor>> Forward(const shared_ptr<Tensor>&) override;
    };
}

#endif //TINYLEARNING_SUM_H
