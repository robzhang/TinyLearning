//
// Created by Fangbo Zhang on 2023/6/3.
//

#ifndef TINYLEARNING_TENSOR_H
#define TINYLEARNING_TENSOR_H

#include <vector>
#include <memory>

using std::vector;
using std::shared_ptr;

const int nMaxTensorAxes = 32;

namespace TinyLearning {
    class Tensor {
    public:
        Tensor() = default;

        explicit Tensor(float value);
        Tensor(const vector<int> &shape, float value);
        Tensor(const vector<int> &shape, const shared_ptr<vector<float>>& data);

        Tensor(const Tensor& tensor);

        Tensor* CloneShape() const;

        Tensor* Reshape(const vector<int> &shape) const;
        Tensor* BroadcastTo(const vector<int>& shape) const;

        Tensor* Transpose(int dim0, int dim1) const;
        Tensor* Transpose(const vector<int>& axes) const;
        vector<int> AxesForTransposingLastTwoDims() const;

        Tensor* Exp();
        Tensor* Tanh();
        Tensor* Sin();
        Tensor* Cos();

        Tensor* Sum(const vector<int>& axis = vector<int>{}, bool keepDims = false);
        Tensor* SumTo(const vector<int>& shape);

        Tensor* Dot(const Tensor *) const;
        Tensor* MatMul(const Tensor *) const;
        Tensor* MatMul(const Tensor &) const;

        inline float DataAt(const vector<int>& axes) const {
            return (*data_)[offset(axes)];
        }

        friend Tensor* operator* (const Tensor &t, float f);
        friend Tensor* operator*(float f, const Tensor &tensor);
        friend Tensor* operator*(const Tensor& tensor1, const Tensor &tensor2);
        friend Tensor* operator+(const Tensor &tensor, float f);
        friend Tensor* operator+(float f, const Tensor &tensor);
        friend Tensor* operator+(const Tensor &tensor1, const Tensor &tensor2);
        friend Tensor* operator- (const Tensor &t, float f);
        friend Tensor* operator- (const Tensor &, const Tensor &);
        friend Tensor& operator-= (Tensor &t, float f);
        friend Tensor& operator-= (Tensor &, const Tensor &);
        friend Tensor* operator/ (const Tensor &t, float f);
        friend Tensor* operator/ (const Tensor &t, const Tensor &);
        friend Tensor* operator^ (const Tensor &t, float c);

        static Tensor* Zeros(const vector<int> &shape);
        static Tensor* Ones(const vector<int> &shape);
        static Tensor* WithValue(const vector<int> &shape, float value);

        static shared_ptr<Tensor> MatMul(const shared_ptr<Tensor> &tensor1, const shared_ptr<Tensor> &tensor2);

        inline bool isShapeSame(const Tensor& other) const {
            return this->shape_ == other.shape_;
        }
        inline bool isShapeSame(const Tensor* other) const {
            return this->shape_ == other->shape_;
        }

        inline const vector<int>& Shape() {
            return shape_;
        }
        inline size_t Size() {
            return data_->size();
        }
        inline size_t Ndim() {
            return shape_.size();
        }

        void Print();
        void PrintShape();

    private:
        shared_ptr<vector<float>> data_;
        vector<int> shape_;
        vector<int> strides_;

        void doPrint(vector<int>index, int offsetInShape);

        inline int offset(const vector<int>& axes) const {
            int nDim = int(shape_.size());
            int offset = 0;

            assert(nDim == axes.size());

            for(int i = 0; i < nDim; ++i) {
                offset += (axes[i] * strides_[i]);
            }

            return offset;
        }

        inline int dimSize() const {
            return static_cast<int>(shape_.size());
        }

        static inline int dimSize(const vector<int>& shape) {
            return static_cast<int>(shape.size());
        }

        void reshape(const vector<int>& shape);

        void validateBroadcastShape(const vector<int> &shape) const;

        int getBroadcastOffset(const vector<int> &axes, const vector<int> &shape) const;

        void computeStrides();

        Tensor* sumAll(bool keepDims) const;
        Tensor* sumAxis(int axis) const;
        Tensor* doSumAxis() const;

        Tensor* matrixMultiply(const Tensor* other) const;
        Tensor* batchedMatrixMultiply(const Tensor* other) const;

        void squeeze(const vector<int>& axes);

        static void broadcastTensors(const Tensor** t1, const Tensor** t2);

        static vector<int>computeBroadcastShape(const vector<int>& shape1, const vector<int>& shape2);

        static void updateAxis(vector<int>& axes, int right, const vector<int>& shape);

        static int countByShape(const vector<int>& shape);

    };

    shared_ptr<Tensor> operator*(const shared_ptr<Tensor> &tensor, float f);
    shared_ptr<Tensor> operator*(float f, const shared_ptr<Tensor> &tensor);
    shared_ptr<Tensor> operator*(const shared_ptr<Tensor> &tensor1, const shared_ptr<Tensor> &tensor2);
    shared_ptr<Tensor> operator+(const shared_ptr<Tensor> &tensor, float f);
    shared_ptr<Tensor> operator+(float f, const shared_ptr<Tensor> &tensor);
    shared_ptr<Tensor> operator+(const shared_ptr<Tensor> &tensor1, const shared_ptr<Tensor> &tensor2);
    shared_ptr<Tensor> operator-(const shared_ptr<Tensor> &tensor, float f);
    shared_ptr<Tensor> operator-(const shared_ptr<Tensor> &tensor1, const shared_ptr<Tensor> &tensor2);
    shared_ptr<Tensor> operator/(const shared_ptr<Tensor> &tensor, float f);
    shared_ptr<Tensor> operator/(const shared_ptr<Tensor> &tensor1, const shared_ptr<Tensor> &tensor2);
    shared_ptr<Tensor> operator^(const shared_ptr<Tensor> &tensor, float c);

    shared_ptr<Tensor> operator-=(const shared_ptr<Tensor> &tensor1, const shared_ptr<Tensor> &tensor2);
}

#endif //TINYLEARNING_TENSOR_H
