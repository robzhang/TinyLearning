//
// Created by Fangbo Zhang on 2023/6/5.
//

#ifndef TINYLEARNING_FUNCTION_H
#define TINYLEARNING_FUNCTION_H

#include <memory>

#include "variable.h"

using namespace std;

namespace TinyLearning {
    class Function : public std::enable_shared_from_this<Function> {
    public:
        virtual ~Function() = default;

        vector<shared_ptr<Variable>> operator()(const shared_ptr<Variable>& input);
        vector<shared_ptr<Variable>> operator()(const shared_ptr<Variable>& input0, const shared_ptr<Variable>& input1);
        vector<shared_ptr<Variable>> operator()(const shared_ptr<Variable>& input0, const shared_ptr<Variable>& input1, const shared_ptr<Variable>& input2);

        virtual vector<shared_ptr<Variable>> Backward(const vector<shared_ptr<Variable>>&) = 0;

        virtual const char* Name() const = 0 ;

        bool operator< (const Function& right) const;

        inline const vector<shared_ptr<Variable>>& Input() const {
            return inputs_;
        }
        inline const vector<weak_ptr<Variable>>& Output() const {
            return outputs_;
        }
        inline int Generation() const {
            return generation_;
        }

        void ClearOutputsGrad() const;

        struct Comparator {
            bool operator()(const shared_ptr<Function>& f1, const shared_ptr<Function>& f2) const {
                return *f1 < *f2;
            }
        };

    private:
        vector<shared_ptr<Variable>> inputs_;
        vector<weak_ptr<Variable>> outputs_;

        int generation_;

        virtual vector<shared_ptr<Tensor>> Forward(const shared_ptr<Tensor>&);
        virtual vector<shared_ptr<Tensor>> Forward(const shared_ptr<Tensor>&, const shared_ptr<Tensor>&);
        virtual vector<shared_ptr<Tensor>> Forward(const shared_ptr<Tensor>&, const shared_ptr<Tensor>&, const shared_ptr<Tensor>&);
    };
}

#endif //TINYLEARNING_FUNCTION_H
