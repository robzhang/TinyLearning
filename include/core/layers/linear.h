//
// Created by Fangbo Zhang on 2023/6/17.
//

#ifndef TINYLEARNING_LINEAR_H
#define TINYLEARNING_LINEAR_H

#include <memory>

#include "../module.h"

namespace TinyLearning {
    class Linear : public Module {
    public:
        Linear(int inSize, int outSize);

        vector<shared_ptr<Variable>> Parameters() override;

    private:
        shared_ptr<Variable> W_;
        shared_ptr<Variable> b_;

        shared_ptr<Variable> Forward(const shared_ptr<Variable>&) override;
    };
}

#endif //TINYLEARNING_LINEAR_H
