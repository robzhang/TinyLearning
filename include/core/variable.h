//
// Created by Fangbo Zhang on 2023/6/5.
//

#ifndef TINYLEARNING_VARIABLE_H
#define TINYLEARNING_VARIABLE_H

#include <utility>
#include <string>

#include "tensor/tensor.h"

using namespace std;

namespace TinyLearning {
    class Function;
    class Variable {
    public:
        Variable() = delete;
        explicit Variable(shared_ptr<Tensor> data, string  name="");

        void backward(bool retain_grad = false);

        void SetCreator(shared_ptr<Function> creator);

        inline shared_ptr<Tensor> Data() {
            return data_;
        }
        inline shared_ptr<Variable> Grad() {
            return grad_;
        }
        inline void SetGrad(shared_ptr<Variable> grad) {
            grad_ = std::move(grad);
        }
        inline void ClearGrad() {
            grad_ = nullptr;
        }
        inline int Generation() const {
            return generation_;
        }
        inline vector<int>Shape() const {
            return data_->Shape();
        }
        inline size_t Size() const {
            return data_->Size();
        }
        inline size_t Ndim() const {
            return data_->Ndim();
        }
        inline string Name() const {
            return name_;
        }
        inline void SetName(const string& name) {
            name_ = name;
        }
        inline shared_ptr<Function> Creator() const {
            return creator_;
        }
        inline void Print() {
            data_->Print();
        }
        inline void PrintShape() {
            data_->PrintShape();
        }

        static shared_ptr<Variable> NewF(float f, const string& name="");
        static shared_ptr<Variable>New(const vector<int>& shape, const vector<float>& data, const string& name="");
        static shared_ptr<Variable>Zeros(const vector<int>& shape, const string& name="");

    private:
        shared_ptr<Tensor> data_;
        shared_ptr<Variable> grad_;
        shared_ptr<Function> creator_;

        int generation_;
        string name_;
    };

    shared_ptr<Variable> operator*(const shared_ptr<Variable>& x0, const shared_ptr<Variable>& x1);
    shared_ptr<Variable> operator*(const shared_ptr<Variable>& x, float f);
    shared_ptr<Variable> operator*(float f, const shared_ptr<Variable>& x);

    shared_ptr<Variable> operator+(const shared_ptr<Variable>& x0, const shared_ptr<Variable>& x1);
    shared_ptr<Variable> operator+(const shared_ptr<Variable>& x, float f);
    shared_ptr<Variable> operator+(float f, const shared_ptr<Variable>& x);

    shared_ptr<Variable> operator-(const shared_ptr<Variable>& x0, const shared_ptr<Variable>& x1);
    shared_ptr<Variable> operator-(const shared_ptr<Variable>& x, float f);
    shared_ptr<Variable> operator-(float f, const shared_ptr<Variable>& x);

    shared_ptr<Variable> operator/(const shared_ptr<Variable>& x0, const shared_ptr<Variable>& x1);
    shared_ptr<Variable> operator/(const shared_ptr<Variable>& x, float f);
    shared_ptr<Variable> operator/(float f, const shared_ptr<Variable>& x);

    shared_ptr<Variable> operator-(const shared_ptr<Variable>& x);

    shared_ptr<Variable> operator^(const shared_ptr<Variable>& x, float c);
}

#endif //TINYLEARNING_VARIABLE_H
