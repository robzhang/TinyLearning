//
// Created by Fangbo Zhang on 2023/6/7.
//

#ifndef TINYLEARNING_SUB_H
#define TINYLEARNING_SUB_H

#include <memory>

#include "../function.h"

namespace TinyLearning {
    class Sub : public Function {
    public:
        vector<shared_ptr<Variable>> Backward(const vector<shared_ptr<Variable>>&) override;

        const char* Name() const override {
            return "Sub";
        }

        static shared_ptr<Sub>New() {
            auto sub = new Sub;
            return shared_ptr<Sub>(sub);
        }

    private:
        Sub() = default;

        vector<shared_ptr<Tensor>> Forward(const shared_ptr<Tensor>&,const shared_ptr<Tensor>&) override;
    };
}

#endif //TINYLEARNING_SUB_H
