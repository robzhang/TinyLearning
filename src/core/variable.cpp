//
// Created by Fangbo Zhang on 2023/6/5.
//
#include <utility>
#include <set>
//#include <iostream>
#include <string>

#include "core/variable.h"
#include "core/function.h"
#include "core/functions.h"

using namespace std;

namespace TinyLearning {
    Variable::Variable(shared_ptr<Tensor> data, string  name) : data_(std::move(data)), generation_(0), name_(std::move(name)) {

    }

    shared_ptr<Variable> Variable::New(const vector<int>& shape, const vector<float>& data, const string& name) {
        auto t = make_shared<Tensor>(shape, make_shared<vector<float>>(data));
        return make_shared<Variable>(t, name);
    }
    shared_ptr<Variable> Variable::NewF(float f, const string& name) {
        auto t = make_shared<Tensor>(f);
        return make_shared<Variable>(t, name);
    }
    shared_ptr<Variable>Variable::Zeros(const vector<int>& shape, const string& name) {
        auto t = shared_ptr<Tensor>(Tensor::Zeros(shape));
        return make_shared<Variable>(t, name);
    }

    void Variable::SetCreator(shared_ptr<Function> creator) {
        creator_ = std::move(creator);

        generation_ = creator_->Generation() + 1;
        //cout << "variable generation:" << generation_ << endl;
    }

    void Variable::backward(bool retain_grad) {
        if (!this->grad_) {
            //this->grad_ = shared_ptr<Tensor>(Tensor::Ones(this->data_->shape()));
            this->grad_ = make_shared<Variable>(shared_ptr<Tensor>(Tensor::Ones(this->data_->Shape())));
        }

        assert(this->creator_);

        std::set<shared_ptr<Function>, Function::Comparator> funcs = {this->creator_};

        do {
            const auto f = *funcs.begin();
            //std::cout << "----pop:" << f->Generation() << endl;
            funcs.erase(f);

            assert(f);

            vector<shared_ptr<Variable>> gys;
            for (const auto& y:f->Output()) {
                gys.push_back(y.lock()->Grad());
            }

            auto gxs = f->Backward(gys);

            auto inputs = f->Input();
            assert(gxs.size() == inputs.size());

            for (int i = 0; i < inputs.size(); i++) {
                auto x = inputs[i];
                if (!x->Grad()) {
                    x->SetGrad(gxs[i]);
                } else {
                    x->SetGrad(gxs[i] + x->Grad());
                }

                if (x->creator_) {
                    //std::cout << "----push:" << x->creator_->Generation() << endl;
                    funcs.insert(x->creator_);
                }
                //std::cout << "funcs size:" << funcs.size() << endl;
            }
            if (!retain_grad) {
                f->ClearOutputsGrad();
            }
        } while (!funcs.empty());
    }

    shared_ptr<Variable> operator*(const shared_ptr<Variable>& x0, const shared_ptr<Variable>& x1) {
        return mul(x0, x1);
    }
    shared_ptr<Variable> operator*(const shared_ptr<Variable>& x, float f) {
        return mul(x, Variable::NewF(f));
    }
    shared_ptr<Variable> operator*(float f, const shared_ptr<Variable>& x) {
        return mul(x, Variable::NewF(f));
    }

    shared_ptr<Variable> operator+(const shared_ptr<Variable>& x0, const shared_ptr<Variable>& x1) {
        return add(x0, x1);
    }
    shared_ptr<Variable> operator+(const shared_ptr<Variable>& x, float f) {
        return add(x, Variable::NewF(f));
    }
    shared_ptr<Variable> operator+(float f, const shared_ptr<Variable>& x) {
        return add(x, Variable::NewF(f));
    }

    shared_ptr<Variable> operator-(const shared_ptr<Variable>& x0, const shared_ptr<Variable>& x1){
        return sub(x0, x1);
    }
    shared_ptr<Variable> operator-(const shared_ptr<Variable>& x, float f){
        return sub(x, Variable::NewF(f));
    }
    shared_ptr<Variable> operator-(float f, const shared_ptr<Variable>& x){
        return sub(Variable::NewF(f), x);
    }

    shared_ptr<Variable> operator/(const shared_ptr<Variable>& x0, const shared_ptr<Variable>& x1){
        return div(x0, x1);
    }
    shared_ptr<Variable> operator/(const shared_ptr<Variable>& x, float f){
        return div(x, Variable::NewF(f));
    }
    shared_ptr<Variable> operator/(float f, const shared_ptr<Variable>& x){
        return div(Variable::NewF(f), x);
    }

    shared_ptr<Variable> operator-(const shared_ptr<Variable>& x){
        return neg(x);
    }

    shared_ptr<Variable> operator^(const shared_ptr<Variable>& x, float c){
        return pow(x, c);
    }
}