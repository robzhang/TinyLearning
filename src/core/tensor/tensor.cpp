//
// Created by Fangbo Zhang on 2023/6/3.
//

#include <vector>
#include <memory>
#include <iostream>
#include <cmath>

#include "core/tensor/tensor.h"

using namespace std;

namespace TinyLearning {
    Tensor* Tensor::Zeros(const vector<int> &shape) {
        return WithValue(shape, 0);
    }
    Tensor* Tensor::Ones(const vector<int> &shape) {
        return WithValue(shape, 1);
    }
    Tensor* Tensor::WithValue(const vector<int> &shape, float value){
        auto tensor = new Tensor(shape, value);
        return tensor;
    }

    Tensor::Tensor(float value) {
        vector<int>shape{1};

        data_ = make_shared<vector<float>>(1, value);
        reshape(shape);
    }

    Tensor::Tensor(const vector<int> &shape, float value) {
        int count = countByShape(shape);
        assert(count > 0);

        data_ = make_shared<vector<float>>(count, value);
        reshape(shape);
    }

    Tensor::Tensor(const vector<int> &shape, const shared_ptr<vector<float>>& data) {
        data_ = data;
        reshape(shape);
    }

    Tensor::Tensor(const Tensor& tensor) {
        data_ = tensor.data_;
        shape_ = tensor.shape_;
        strides_ = tensor.strides_;
    }

    void Tensor::reshape(const vector<int>& shape) {
        assert(dimSize(shape) <= nMaxTensorAxes);

        assert(countByShape(shape) == data_->size());

        shape_ = shape;

        computeStrides();
    }

    Tensor* Tensor::Reshape(const vector<int>& shape) const {
        auto t = new Tensor(*this);

        t->reshape(shape);

        return t;
    }

    Tensor* Tensor::BroadcastTo(const vector<int> &shape) const {
        validateBroadcastShape(shape);

        int target_size = countByShape(shape);

        auto data = new vector<float>(target_size);
        vector<int> axes(dimSize(shape), 0);

        for (int i = 0; i < target_size; ++i) {
            int off = getBroadcastOffset(axes, shape);

            (*data)[i] = (*this->data_)[off];

            updateAxis(axes, int(axes.size()) - 1, shape);
        }

        auto *t = new Tensor();

        t->data_ = shared_ptr<vector<float>>(data);

        t->reshape(shape);

        return t;
    }

    void Tensor::validateBroadcastShape(const vector<int> &shape) const {
        int off = dimSize(shape) - this->dimSize();

        assert(off >= 0);

        for (int i = 0; i < this->dimSize(); ++i) {
            assert(shape[off + i] == this->shape_[i] || this->shape_[i] == 1);
        }
    }

    int Tensor::getBroadcastOffset(const vector<int> &axes, const vector<int> &shape) const {
        int fromDim = this->dimSize();
        int toDim = dimSize(shape);

        int beginAt = toDim - fromDim;

        int off = 0;
        for (int i = fromDim - 1; i >= 0; --i) {
            int k = beginAt + i;
            int index = (axes[k] % this->shape_[i]);
            off += index * strides_[i];
        }

        return off;
    }

    int Tensor::countByShape(const vector<int>& shape) {
        if (shape.empty()) {
            return 0;
        }

        int count = 1;
        for (int i : shape) {
            count *= i;
        }

        return count;
    }

    void Tensor::computeStrides() {
        int nDim = this->dimSize();
        int stride = 1;

        strides_.resize(nDim);

        for (int i = nDim - 1; i >= 0; --i) {
            strides_[i] = stride;
            stride *= shape_[i];
        }
    }

    Tensor* Tensor::Transpose(int dim0, int dim1) const {
        auto* t = new Tensor(*this);
/*
        if (t->dimSize() == 1) {
            t->reshape(vector<int>{1, t->shape_[0]});
        }
*/
        std::swap(t->shape_[dim0], t->shape_[dim1]);
        std::swap(t->strides_[dim0], t->strides_[dim1]);

        return t;
    }

    Tensor* Tensor::Transpose(const vector<int>& axes) const {
        assert(axes.size() == this->dimSize());

        auto* t = new Tensor(*this);
/*
        if (t->dimSize() == 1) {
            t->reshape(vector<int>{1, t->shape_[0]});
        }
*/
        int nDim = t->dimSize();

        for (int i = 0; i < nDim; ++i) {
            t->shape_[i] = shape_[axes[i]];
            t->strides_[i] = strides_[axes[i]];
        }

        return t;
    }

    vector<int> Tensor::AxesForTransposingLastTwoDims() const {
        auto nDim = this->dimSize();

        assert(nDim >= 2);

        if (nDim == 2) {
            return vector<int>{1,0};
        }

        vector<int>axes(nDim);
        for (auto i = 0; i < nDim - 2; ++i) {
            axes[i] = i;
        }

        axes[nDim - 2] = nDim - 1;
        axes[nDim - 1] = nDim - 2;

        return axes;
    }

    Tensor* operator*(const Tensor &tensor, float f) {
        auto *t = tensor.CloneShape();

        for (int i = 0; i < tensor.data_->size(); i++) {
            (*t->data_)[i] = (*tensor.data_)[i] * f;
        }

        return t;
    }
    shared_ptr<Tensor> operator*(const shared_ptr<Tensor> &tensor, float f) {
        auto t = (*tensor) * f;

        return shared_ptr<Tensor>(t);
    }

    Tensor* operator*(float f, const Tensor &tensor) {
        return tensor * f;
    }
    shared_ptr<Tensor> operator*(float f, const shared_ptr<Tensor> &tensor) {
        auto t = (*tensor) * f;

        return shared_ptr<Tensor>(t);
    }

    Tensor* operator*(const Tensor &tensor1, const Tensor &tensor2) {
        auto x1 = &tensor1;
        auto x2 = &tensor2;

        Tensor::broadcastTensors(&x1, &x2);

        auto *t = x1->CloneShape();

        for (int i = 0; i < x1->data_->size(); i++) {
            (*t->data_)[i] = (*x1->data_)[i] * (*x2->data_)[i];
        }

        return t;
    }
    shared_ptr<Tensor> operator*(const shared_ptr<Tensor> &tensor1, const shared_ptr<Tensor> &tensor2) {
        auto t = (*tensor1) * (*tensor2);

        return shared_ptr<Tensor>(t);
    }

    Tensor* operator+(const Tensor &tensor, float f) {
        auto *t = tensor.CloneShape();

        for (int i = 0; i < tensor.data_->size(); i++) {
            (*t->data_)[i] = (*tensor.data_)[i] + f;
        }

        return t;
    }
    shared_ptr<Tensor> operator+(const shared_ptr<Tensor> &tensor, float f) {
        auto t = *tensor + f;

        return shared_ptr<Tensor>(t);
    }
    Tensor* operator+(float f, const Tensor &tensor) {
        return tensor + f;
    }
    shared_ptr<Tensor> operator+(float f, const shared_ptr<Tensor> &tensor) {
        auto t = *tensor + f;

        return shared_ptr<Tensor>(t);
    }
    Tensor* operator+(const Tensor &tensor1, const Tensor &tensor2) {
        auto x1 = &tensor1;
        auto x2 = &tensor2;

        Tensor::broadcastTensors(&x1, &x2);

        auto *t = x1->CloneShape();

        for (int i = 0; i < x1->data_->size(); i++) {
            (*t->data_)[i] = (*x1->data_)[i] + (*x2->data_)[i];
        }

        return t;
    }
    shared_ptr<Tensor> operator+(const shared_ptr<Tensor> &tensor1, const shared_ptr<Tensor> &tensor2) {
        auto t = *tensor1 + *tensor2;

        return shared_ptr<Tensor>(t);
    }

    Tensor* operator-(const Tensor &tensor, float f) {
        auto *t = tensor.CloneShape();

        for (int i = 0; i < tensor.data_->size(); i++) {
            (*t->data_)[i] = (*tensor.data_)[i] - f;
        }

        return t;
    }
    shared_ptr<Tensor> operator-(const shared_ptr<Tensor> &tensor, float f) {
        auto t = *tensor - f;

        return shared_ptr<Tensor>(t);
    }

    Tensor* operator-(const Tensor &tensor1, const Tensor &tensor2) {
        auto x1 = &tensor1;
        auto x2 = &tensor2;

        Tensor::broadcastTensors(&x1, &x2);

        auto *t = x1->CloneShape();

        for (int i = 0; i < x1->data_->size(); i++) {
            (*t->data_)[i] = (*x1->data_)[i] - (*x2->data_)[i];
        }

        return t;
    }
    shared_ptr<Tensor> operator-(const shared_ptr<Tensor> &tensor1, const shared_ptr<Tensor> &tensor2) {
        auto t = *tensor1 - *tensor2;

        return shared_ptr<Tensor>(t);
    }

    Tensor& operator-=(Tensor &tensor, float f) {
        for (int i = 0; i < tensor.data_->size(); i++) {
            (*tensor.data_)[i] -= f;
        }

        return tensor;
    }
    shared_ptr<Tensor> operator-=(shared_ptr<Tensor> &tensor, float f) {
        *tensor -= f;

        return tensor;
    }

    Tensor& operator-=(Tensor &tensor1, const Tensor &tensor2) {
        assert(tensor1.isShapeSame(tensor2));

        for (int i = 0; i < tensor1.data_->size(); i++) {
            (*tensor1.data_)[i] -= (*tensor2.data_)[i];
        }

        return tensor1;
    }
    shared_ptr<Tensor> operator-=(const shared_ptr<Tensor> &tensor1, const shared_ptr<Tensor> &tensor2) {
        *tensor1 -= *tensor2;

        return tensor1;
    }

    Tensor* operator/(const Tensor &tensor, float f) {
        auto *t = tensor.CloneShape();

        for (int i = 0; i < tensor.data_->size(); i++) {
            (*t->data_)[i] = (*tensor.data_)[i] / f;
        }

        return t;
    }
    shared_ptr<Tensor> operator/(const shared_ptr<Tensor> &tensor, float f) {
        auto t = (*tensor) / f;

        return shared_ptr<Tensor>(t);
    }
    Tensor* operator/(const Tensor &tensor1, const Tensor &tensor2) {
        auto x1 = &tensor1;
        auto x2 = &tensor2;

        Tensor::broadcastTensors(&x1, &x2);

        auto *t = x1->CloneShape();

        for (int i = 0; i < x1->data_->size(); i++) {
            (*t->data_)[i] = (*x1->data_)[i] / (*x2->data_)[i];
        }

        return t;
    }
    shared_ptr<Tensor> operator/(const shared_ptr<Tensor> &tensor1, const shared_ptr<Tensor> &tensor2) {
        auto t = (*tensor1) / (*tensor2);

        return shared_ptr<Tensor>(t);
    }

    Tensor* operator^(const Tensor &tensor, float c) {
        auto *t = tensor.CloneShape();

        for (int i = 0; i < tensor.data_->size(); i++) {
            (*t->data_)[i] = std::powf((*tensor.data_)[i], c);
        }

        return t;
    }
    shared_ptr<Tensor> operator^(const shared_ptr<Tensor> &tensor, float c) {
        auto t = (*tensor) ^ c;

        return shared_ptr<Tensor>(t);
    }

    Tensor* Tensor::Dot(const Tensor* other) const {
        assert(this->dimSize() == 1 && other->dimSize() == 1 && this->shape_[0] == other->shape_[0]);

        float dot = 0;
        for (auto i = 0; i < this->shape_[0]; ++i) {
            dot += (*this->data_)[i] * (*other->data_)[i];
        }

        return new Tensor(dot);
    }

    Tensor* Tensor::MatMul(const Tensor& other) const {
        return this->MatMul(&other);
    }

    Tensor* Tensor::MatMul(const Tensor* other) const {
        if (this->dimSize() == 1 && this->dimSize() == 1) {
            return this->Dot(other);
        }

        if (this->dimSize() == 2 && other->dimSize() == 2) {
            return this->matrixMultiply(other);
        }

        if (this->dimSize() == 1 && other->dimSize() == 2) {
            auto tmp = this->Reshape(vector<int>{1, this->shape_[0]});

            auto t = tmp->matrixMultiply(other);

            t->reshape(vector<int>{t->shape_[1]});

            return t;
        }

        if (this->dimSize() == 2 && other->dimSize() == 1) {
            auto tmp = other->Reshape(vector<int>{other->shape_[0], 1});

            auto t = this->matrixMultiply(tmp);

            t->reshape(vector<int>{t->shape_[0]});

            return t;
        }

        return this->batchedMatrixMultiply(other);
    }

    shared_ptr<Tensor> Tensor::MatMul(const shared_ptr<Tensor> &tensor1, const shared_ptr<Tensor> &tensor2) {
        auto t = (*tensor1).MatMul(*tensor2);

        return shared_ptr<Tensor>(t);
    }

    Tensor* Tensor::matrixMultiply(const Tensor* other) const {
        assert(this->dimSize() == 2 && other->dimSize() == 2 && this->shape_[1] == other->shape_[0]);

        int r = this->shape_[0], c = other->shape_[1];
        auto data = new vector<float>(r * c, 0);
        for (auto i = 0; i < r; ++i) {
            auto base = i * c;
            for (auto j = 0; j < c; ++j) {
                auto off = base + j;
                for (auto n = 0; n < this->shape_[1]; ++n) {
                    (*data)[off] += this->DataAt(vector<int>{i, n}) * other->DataAt(vector<int>{n, j});
                }
            }
        }

        auto *t = new Tensor();

        t->data_ = shared_ptr<vector<float>>(data);

        t->reshape(vector<int>{r, c});

        return t;
    }

    Tensor* Tensor::batchedMatrixMultiply(const Tensor* other) const {
        throw std::runtime_error("batchedMatrixMultiply NOT Implemented!");

        return this->matrixMultiply(other);
    }

    Tensor* Tensor::Exp() {
        auto *t = this->CloneShape();

        for (int i = 0; i < this->data_->size(); i++) {
            (*t->data_)[i] = std::expf((*this->data_)[i]);
        }

        return t;
    }
    Tensor* Tensor::Tanh() {
        auto *t = this->CloneShape();

        for (int i = 0; i < this->data_->size(); i++) {
            auto x = (*this->data_)[i];

            auto ex = std::expf(x);
            auto e_x = std::expf(-x);

            (*t->data_)[i] = (ex - e_x) / (ex + e_x);
        }

        return t;
    }
    Tensor* Tensor::Sin() {
        auto *t = this->CloneShape();

        for (int i = 0; i < this->data_->size(); i++) {
            (*t->data_)[i] = std::sin((*this->data_)[i]);
        }

        return t;
    }
    Tensor* Tensor::Cos() {
        auto *t = this->CloneShape();

        for (int i = 0; i < this->data_->size(); i++) {
            (*t->data_)[i] = std::cos((*this->data_)[i]);
        }

        return t;
    }

    Tensor* Tensor::Sum(const vector<int>& axes, bool keepDims) {
        // only support 1 axis at most
        assert(static_cast<int>(axes.size()) <= this->dimSize());

        if (axes.empty() || this->dimSize() == 1) {
            return sumAll(keepDims);
        }

        Tensor* t = this;
        for (auto axis : axes) {
            t = t->sumAxis(axis);
        }

        if (!keepDims) {
            t->squeeze(axes);
        }

        return t;
    }

    Tensor* Tensor::sumAll(bool keepDims) const {
        auto shape = !keepDims ? vector<int>{1} : vector<int>(this->dimSize(), 1);

        auto t = Zeros(shape);

        float sum = 0;
        for (auto d : *data_) {
            sum += d;
        }

        (*t->data_)[0] = sum;

        return t;
    }

    Tensor* Tensor::sumAxis(int axis) const {
        assert(axis >= 0 && axis < dimSize());

        auto xt = this->Transpose(dimSize() - 1, axis);

        auto t = xt->doSumAxis();

        return t->Transpose(dimSize() - 1, axis);
    }

    Tensor* Tensor::doSumAxis() const {
        auto shape = this->shape_;
        shape[dimSize(shape) - 1] = 1;

        auto axes = vector<int>(this->dimSize(), 0);

        auto t = Zeros(shape);
        auto n = this->data_->size() / this->shape_.back();

        for (int i = 0; i < n; i++) {
            float sum = 0;
            for (int j = 0; j < this->shape_.back(); j++) {
                axes[axes.size() - 1] = j;
                sum += this->DataAt(axes);
            }
            //cout << "sum:" << sum << endl;

            axes[axes.size() - 1] = 0;
            (*t->data_)[t->offset(axes)] = sum;

            updateAxis(axes, int(axes.size()) - 2, this->shape_);
        }

        return t;
    }

    void Tensor::squeeze(const vector<int>& axes) {
        for (auto axis : axes) {
            this->shape_[axis] = 0;
        }

        int i = 0;
        for (int j = 0; j < this->dimSize(); ++j) {
            if (this->shape_[j] != 0) {
                if (i != j) {
                    this->shape_[i] = this->shape_[j];
                    this->strides_[i] = this->strides_[j];
                }

                ++i;
            }
        }

        if (i != 0) {
            this->shape_.resize(i);
        } else {
            this->shape_[0] = 1;
            this->shape_.resize(1);
        }
    }

    void Tensor::updateAxis(vector<int>& axes, int right, const vector<int>& shape) {
        for (int i = right; i >= 0; --i) {
            axes[i] += 1;
            if (axes[i] < shape[i]) {
                return;
            } else {
                axes[i] = 0;
            }
        }
    }

    Tensor* Tensor::CloneShape() const {
        auto *t = new Tensor();
        t->data_ = make_shared<vector<float>>(this->data_->size());
        t->shape_ = this->shape_;
        t->strides_ = this->strides_;

        return t;
    }

    Tensor* Tensor::SumTo(const vector<int>& shape) {
        auto nDim = dimSize(shape);
        auto lead = this->dimSize() - nDim;
        assert(lead >= 0);

        vector<int> axes(this->dimSize());
        for (auto i = 0; i< lead; ++i) {
            axes[i] = i;
        }

        auto count = lead;
        for (auto i = 0; i < dimSize(shape); ++i) {
            if (shape[i] == 1) {
                axes[count++] = lead + i;
            }
        }

        axes.resize(count);

        auto y = this->Sum(axes, true);
        if (lead > 0) {
            axes.resize(lead);
            y->squeeze(axes);
        }

        return y;
    }

    void Tensor::broadcastTensors(const Tensor** t1, const Tensor** t2) {
        if((*t1)->isShapeSame(*t2)) {
            return;
        }

        auto tensor1 = *t1;
        auto tensor2 = *t2;

        auto shape = Tensor::computeBroadcastShape(tensor1->shape_, tensor2->shape_);
        if (tensor1->shape_ != shape) {
            *t1 = tensor1->BroadcastTo(shape);
        }
        if (tensor2->shape_ != shape) {
            *t2 = tensor2->BroadcastTo(shape);
        }
    }

    vector<int>Tensor::computeBroadcastShape(const vector<int>& shape1, const vector<int>& shape2) {
        int maxDim = std::max(int(shape1.size()), int(shape2.size()));

        auto padding1 = maxDim - int(shape1.size());
        auto padding2 = maxDim - int(shape2.size());

        vector<int> shape(maxDim);
        for (auto i = maxDim - 1; i >= 0; --i) {
            int v1 = i >= padding1 ? shape1[i - padding1] : 1;
            int v2 = i >= padding2 ? shape2[i - padding2] : 1;

            //cout << i << "---v1: " << v1 << ", v2:" << v2 << endl;

            assert(v1 == v2 || (v1 == 1 || v2 == 1));

            shape[i] = std::max(v1, v2);
        }

        return shape;
    }

    void Tensor::Print() {
        vector<int>index = {};

        doPrint(index, 0);
    }
    void Tensor::doPrint(vector<int>index, int offsetInShape) {
        if (offsetInShape < this->dimSize() - 1) {
            for (int i = 0; i < this->shape_[offsetInShape]; ++i) {
                index.push_back(i);
                this->doPrint(index, offsetInShape+1);
                index.pop_back();
            }
        } else {
            for (int i = 0; i < this->shape_[offsetInShape]; ++i) {
                index.push_back(i);
                auto value = this->DataAt(index);
                cout << value << ' ';
                index.pop_back();
            }
            cout << endl;
        }
    }
    void Tensor::PrintShape() {
        cout << "shape: ";
        for (auto v : this->shape_) {
            cout << v << ' ';
        }
        cout << endl;
    }
}