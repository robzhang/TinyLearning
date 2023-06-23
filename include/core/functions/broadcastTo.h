//
// Created by Fangbo Zhang on 2023/6/12.
//

#ifndef TINYLEARNING_BROADCASTTO_H
#define TINYLEARNING_BROADCASTTO_H

#include <memory>
#include <utility>

#include "../function.h"

namespace TinyLearning {
    class BroadcastTo : public Function {
    public:

        vector<shared_ptr<Variable>> Backward(const vector<shared_ptr<Variable>>&) override;

        const char* Name() const override {
            return "BroadcastTo";
        }

        static shared_ptr<BroadcastTo>New(vector<int> y_shape) {
            auto broadcastTo = new BroadcastTo(std::move(y_shape));
            return shared_ptr<BroadcastTo>(broadcastTo);
        }

    private:
        explicit BroadcastTo(vector<int> y_shape) : y_shape_(std::move(y_shape)){}

        vector<int> x_shape_;
        vector<int> y_shape_;

        vector<shared_ptr<Tensor>> Forward(const shared_ptr<Tensor>&) override;
    };
}

#endif //TINYLEARNING_BROADCASTTO_H
