//
// Created by Fangbo Zhang on 2023/6/10.
//

#ifndef TINYLEARNING_TRANSPOSE_H
#define TINYLEARNING_TRANSPOSE_H

#include <memory>
#include <utility>

#include "../function.h"

namespace TinyLearning {
    class Transpose : public Function {
    public:
        vector<shared_ptr<Variable>> Backward(const vector<shared_ptr<Variable>>&) override;

        const char* Name() const override {
            return "Transpose";
        }

        static shared_ptr<Transpose>New(vector<int> axes) {
            auto transpose = new Transpose(std::move(axes));
            return shared_ptr<Transpose>(transpose);
        }

    private:
        explicit Transpose(vector<int> axes) : axes_(std::move(axes)){}

        static vector<int> inverseTransposeAxes(const vector<int>&);

        vector<int> axes_;

        vector<shared_ptr<Tensor>> Forward(const shared_ptr<Tensor>&) override;
    };
}

#endif //TINYLEARNING_TRANSPOSE_H
