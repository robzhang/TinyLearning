//
// Created by Fangbo Zhang on 2023/6/15.
//

#ifndef TINYLEARNING_MATMUL_H
#define TINYLEARNING_MATMUL_H

#include <memory>

#include "../function.h"

namespace TinyLearning {
    class MatMul : public Function {
    public:
        vector<shared_ptr<Variable>> Backward(const vector<shared_ptr<Variable>>&) override;

        const char* Name() const override {
            return "MatMul";
        }

        static shared_ptr<MatMul>New() {
            auto square = new MatMul;
            return shared_ptr<MatMul>(square);
        }

    private:
        MatMul() = default;

        vector<shared_ptr<Tensor>> Forward(const shared_ptr<Tensor>&,const shared_ptr<Tensor>&) override;
    };
}

#endif //TINYLEARNING_MATMUL_H
