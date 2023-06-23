//
// Created by Fangbo Zhang on 2023/6/5.
//
#include <memory>
#include <iostream>

#include "core/variable.h"
#include "core/function.h"

using namespace std;

namespace TinyLearning {
    vector<shared_ptr<Variable>> Function::operator()(const shared_ptr<Variable>& input) {
        this->generation_ = input->Generation();
        //cout << "function generation:" << generation_ << endl;

        auto x = input->Data();

        auto y = Forward(x);

        vector<shared_ptr<Variable>>outputs;
        for (const auto& o: y) {
            auto output = make_shared<Variable>(o);

            output->SetCreator(shared_from_this());

            this->output_.push_back(weak_ptr<Variable>(output));
            outputs.push_back(output);
        }

        this->input_ = vector<shared_ptr<Variable>>{input};

        return outputs;
    }
    vector<shared_ptr<Variable>> Function::operator()(const shared_ptr<Variable>& input0, const shared_ptr<Variable>& input1) {
        //vector<shared_ptr<Variable>>bag = ;
        this->generation_ = std::max(input0->Generation(), input1->Generation());
        //cout << "function generation:" << generation_ << endl;

        auto x0 = input0->Data();
        auto x1 = input1->Data();

        auto y = Forward(x0, x1);

        vector<shared_ptr<Variable>>outputs;
        for (const auto& o: y) {
            auto output = make_shared<Variable>(o);

            output->SetCreator(shared_from_this());

            this->output_.push_back(output);
            outputs.push_back(output);
        }

        this->input_ = vector<shared_ptr<Variable>>{input0, input1};

        return outputs;
    }
    vector<shared_ptr<Variable>> Function::operator()(const shared_ptr<Variable>& input0, const shared_ptr<Variable>& input1, const shared_ptr<Variable>& input2) {
        this->generation_ = std::max(std::max(input0->Generation(), input1->Generation()), input2->Generation());
        //cout << "function generation:" << generation_ << endl;

        auto x0 = input0->Data();
        auto x1 = input1->Data();
        auto x2 = input2->Data();

        auto y = Forward(x0, x1, x2);

        vector<shared_ptr<Variable>>outputs;
        for (const auto& o: y) {
            auto output = make_shared<Variable>(o);

            output->SetCreator(shared_from_this());

            this->output_.push_back(output);
            outputs.push_back(output);
        }

        this->input_ = vector<shared_ptr<Variable>>{input0, input1, input2};

        return outputs;
    }

    vector<shared_ptr<Tensor>> Function::Forward(const shared_ptr<Tensor>&) {
        throw std::runtime_error("Forward(x) NOT implemented");
    }
    vector<shared_ptr<Tensor>> Function::Forward(const shared_ptr<Tensor>&, const shared_ptr<Tensor>&) {
        throw std::runtime_error("Forward(x,y) NOT implemented");
    }
    vector<shared_ptr<Tensor>> Function::Forward(const shared_ptr<Tensor>&, const shared_ptr<Tensor>&, const shared_ptr<Tensor>&) {
        throw std::runtime_error("Forward(x,y,z) NOT implemented");
    }

    bool Function::operator< (const Function& right) const {
        if (this == &right) {
            //cout << "insert duplicated Function" << endl;
            return false;
        }

        if (right.generation_ != this->generation_) {
            return right.generation_ < this->generation_;
        } else {
            return this < &right; // 没有实际意义，仅仅是为了区分两个不同的Function
        }
    }

    void Function::ClearOutputsGrad() {
        for (const auto& y:this->Output()) {
            y.lock()->ClearGrad();
        }
    }
}