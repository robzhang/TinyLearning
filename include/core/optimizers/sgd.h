//
// Created by Fangbo Zhang on 2023/6/19.
//

#ifndef TINYLEARNING_SGD_H
#define TINYLEARNING_SGD_H

#include "../optimizer.h"

namespace TinyLearning {
    class SGD : public Optimizer {
    public:
        SGD(const vector<shared_ptr<Variable>>& parameters, float learningRate);

    private:
        void updateOne(shared_ptr<Variable>&) override;
    };
}

#endif //TINYLEARNING_SGD_H
