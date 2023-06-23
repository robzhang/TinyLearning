//
// Created by Fangbo Zhang on 2023/6/17.
//

#ifndef TINYLEARNING_MODULE_H
#define TINYLEARNING_MODULE_H

#include "variable.h"

using namespace std;

namespace TinyLearning {
    class Module {
    public:
        template<typename... Args>
        shared_ptr<Variable> operator()(const Args&... args) {
            auto outputs = Forward(args...);

            return outputs;
        }

        virtual vector<shared_ptr<Variable>> Parameters() = 0;

        void clearGrads();

    private:
        virtual shared_ptr<Variable> Forward(const shared_ptr<Variable>&);
        virtual shared_ptr<Variable> Forward(const shared_ptr<Variable>&, const shared_ptr<Variable>&);
        virtual shared_ptr<Variable> Forward(const shared_ptr<Variable>&, const shared_ptr<Variable>&, const shared_ptr<Variable>&);
    };
}

#endif //TINYLEARNING_MODULE_H
