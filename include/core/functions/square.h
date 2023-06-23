//
// Created by Fangbo Zhang on 2023/6/5.
//

#ifndef TINYLEARNING_SQUARE_H
#define TINYLEARNING_SQUARE_H

#include <memory>

#include "../function.h"

namespace TinyLearning {
    class Square : public Function {
    public:
        vector<shared_ptr<Variable>> Backward(const vector<shared_ptr<Variable>>&) override;

        const char* Name() const override {
            return "Square";
        }

        static shared_ptr<Square>New() {
            auto square = new Square;
            return shared_ptr<Square>(square);
        }

    private:
        Square() = default;

        vector<shared_ptr<Tensor>> Forward(const shared_ptr<Tensor>&) override;
    };
}

#endif //TINYLEARNING_SQUARE_H
