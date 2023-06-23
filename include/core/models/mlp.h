//
// Created by Fangbo Zhang on 2023/6/19.
//

#ifndef TINYLEARNING_MLP_H
#define TINYLEARNING_MLP_H

#include <functional>

#include "../module.h"
#include "../functions.h"

using namespace std;

namespace TinyLearning {
    class MLP : public Module {
    public:
        MLP(const std::initializer_list<int> &, const function<shared_ptr<Variable>(const shared_ptr<Variable>&)>& activation = sigmoid);

        vector<shared_ptr<Variable>> Parameters() override;

    private:
        vector<shared_ptr<Module>> layers_;
        function<shared_ptr<Variable>(const shared_ptr<Variable>&)> activation_;
        vector<shared_ptr<Variable>> parameters_;

        shared_ptr<Variable> Forward(const shared_ptr<Variable>&) override;

        void setupParameters();
    };
}

#endif //TINYLEARNING_MLP_H
