//
// Created by Fangbo Zhang on 2023/6/5.
//
#include <memory>
#include <utility>

#include "core/variable.h"
#include "core/functions/square.h"
#include "core/functions/exp.h"
#include "core/functions/add.h"
#include "core/functions/mul.h"
#include "core/functions/neg.h"
#include "core/functions/sub.h"
#include "core/functions/div.h"
#include "core/functions/pow.h"
#include "core/functions/sin.h"
#include "core/functions/cos.h"
#include "core/functions/tanh.h"
#include "core/functions/reshape.h"
#include "core/functions/transpose.h"
#include "core/functions/sum.h"
#include "core/functions/sumTo.h"
#include "core/functions/broadcastTo.h"
#include "core/functions/matMul.h"
#include "core/functions/meanSquaredError.h"
#include "core/functions/linear.h"
#include "core/functions/sigmoid.h"
#include "core/functions.h"

using namespace std;

namespace TinyLearning {
    shared_ptr<Variable> square(const shared_ptr<Variable>& x) {
        auto square = Square::New();
        return (*square)(x)[0];
    }

    shared_ptr<Variable> exp(const shared_ptr<Variable>& x) {
        auto exp = Exp::New();
        return (*exp)(x)[0];
    }

    shared_ptr<Variable> add(const shared_ptr<Variable>& x0, const shared_ptr<Variable>& x1) {
        auto add = Add::New();
        return (*add)(x0, x1)[0];
    }

    shared_ptr<Variable> mul(const shared_ptr<Variable>& x0, const shared_ptr<Variable>& x1) {
        auto mul = Mul::New();
        return (*mul)(x0, x1)[0];
    }

    shared_ptr<Variable> neg(const shared_ptr<Variable>& x) {
        auto neg = Neg::New();
        return (*neg)(x)[0];
    }

    shared_ptr<Variable> sub(const shared_ptr<Variable>& x0, const shared_ptr<Variable>& x1) {
        auto sub = Sub::New();
        return (*sub)(x0, x1)[0];
    }

    shared_ptr<Variable> div(const shared_ptr<Variable>& x0, const shared_ptr<Variable>& x1) {
        auto div = Div::New();
        return (*div)(x0, x1)[0];
    }

    shared_ptr<Variable> pow(const shared_ptr<Variable>& x, float c) {
        auto pow = Pow::New(c);
        return (*pow)(x)[0];
    }

    shared_ptr<Variable> sin(const shared_ptr<Variable>& x) {
        auto sin = Sin::New();
        return (*sin)(x)[0];
    }

    shared_ptr<Variable> cos(const shared_ptr<Variable>& x) {
        auto cos = Cos::New();
        return (*cos)(x)[0];
    }

    shared_ptr<Variable> tanh(const shared_ptr<Variable>& x) {
        auto tanh = Tanh::New();
        return (*tanh)(x)[0];
    }

    shared_ptr<Variable> reshape(const shared_ptr<Variable>& x, const vector<int>& shape) {
        if (x->Shape() == shape) {
            return x;
        }

        auto reshape = Reshape::New(shape);
        return (*reshape)(x)[0];
    }

    shared_ptr<Variable> transpose(const shared_ptr<Variable>& x, const vector<int>& axes) {
        auto transpose = Transpose::New(axes);
        return (*transpose)(x)[0];
    }

    shared_ptr<Variable> sum(const shared_ptr<Variable>& x, const vector<int>& axes, bool keepDims) {
        auto sum = Sum::New(axes, keepDims);
        return (*sum)(x)[0];
    }

    shared_ptr<Variable> sumTo(const shared_ptr<Variable>& x, const vector<int>& shape) {
        if (x->Shape() == shape) {
            return x;
        }

        auto sumTo = SumTo::New(shape);
        return (*sumTo)(x)[0];
    }

    shared_ptr<Variable> broadcastTo(const shared_ptr<Variable>& x, const vector<int>& shape) {
        if (x->Shape() == shape) {
            return x;
        }

        auto broadcastTo = BroadcastTo::New(shape);
        return (*broadcastTo)(x)[0];
    }

    shared_ptr<Variable> matMul(const shared_ptr<Variable>& x, const shared_ptr<Variable>& W) {
        auto matMul = MatMul::New();
        return (*matMul)(x, W)[0];
    }

    shared_ptr<Variable> meanSquaredError(const shared_ptr<Variable>& x0, const shared_ptr<Variable>& x1) {
        auto meanSquaredError = MeanSquaredError::New();
        return (*meanSquaredError)(x0, x1)[0];
    }

    shared_ptr<Variable> linear(const shared_ptr<Variable>& x, const shared_ptr<Variable>& W, const shared_ptr<Variable>& b) {
        auto linear = LinearFunction::New();
        return (*linear)(x, W, b)[0];
    }
    shared_ptr<Variable> sigmoid(const shared_ptr<Variable>& x) {
        auto sigmoid = Sigmoid::New();
        return (*sigmoid)(x)[0];
    }
}
