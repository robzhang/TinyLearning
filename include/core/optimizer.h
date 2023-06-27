//
// Created by Fangbo Zhang on 2023/6/19.
//

#ifndef TINYLEARNING_OPTIMIZER_H
#define TINYLEARNING_OPTIMIZER_H

#include <vector>

#include "variable.h"

using namespace std;

namespace TinyLearning {
    class Optimizer {
    public:
        Optimizer(const vector<shared_ptr<Variable>>& parameters, float learningRate);

        virtual ~Optimizer() = default;

        void Update();
        void ClearGrads();

        inline float LearningRate() const {
            return learningRate_;
        }

    private:
        float learningRate_;
        vector<shared_ptr<Variable>> parameters_;

        virtual void updateOne(shared_ptr<Variable>& parameter) = 0;
    };
}

#endif //TINYLEARNING_OPTIMIZER_H
