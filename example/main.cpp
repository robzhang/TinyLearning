//
// Created by Fangbo Zhang on 2023/6/4.
//
#include <memory>
#include <iostream>
#include <cmath>

#include "core/tensor/tensor.h"
#include "core/variable.h"
#include "core/functions.h"
#include "core/numerical_diff.h"
#include "core/layers/linear.h"
#include "core/models/mlp.h"
#include "core/optimizers/sgd.h"
#include "graphviz/graphviz.h"
#include "utils/utils.h"

using namespace std;

using namespace TinyLearning;

#define PI 3.1415926

void tensorExample() {
    vector<int> shape = {2,3,2};
    shared_ptr<vector<float>> data = make_shared<vector<float>>(vector<float>{1,2,3,4,5,6,7,8,9,10,11,12});

    auto t = new Tensor(shape, data);
    t->Print();
    t->Reshape(vector<int>{2,6});
    t->Print();
    //auto a = t->Tranpose(1,0);
    //a->Print();
    //auto a = 2 + (*t);
    auto a = (*t) ^ 2;
    a->Print();
}

void addExample() {
    /*
    auto x0 = Variable::New(vector<int>{1}, vector<float>{2});
    auto x1 = Variable::New(vector<int>{1}, vector<float>{3});

    auto y = (x0 ^ 2) + (x1 ^ 2);
    y->backward();

    cout << "y is: ";
    y->Print();
    x0->Grad()->Print();
    x1->Grad()->Print();

    x0 = Variable::New(vector<int>{3}, vector<float>{1,2,3});
    x1 = Variable::New(vector<int>{1}, vector<float>{10});

    y = x0 + x1;
    y->Print();

    y->backward();
    x1->Grad()->Print();*/
    auto a = Variable::New(vector<int>{2,3}, vector<float>{0,1,2,3,4,5}, "a");
    auto b = Variable::New(vector<int>{2,3,3}, vector<float>{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17}, "b");

    auto c = reshape(b, vector<int>{3,2,3});
    auto y = a + c;
    y->Print();

    auto d = reshape(b, vector<int>{6,1,3});
    y = d + a;
    y->Print();
}
void mulExample() {
    auto x0 = Variable::New(vector<int>{1}, vector<float>{3});
    auto x1 = Variable::New(vector<int>{1}, vector<float>{2});
    auto x2 = Variable::New(vector<int>{1}, vector<float>{1});

    auto y = x0 * x1 + x2;

    y->backward();

    cout << "y is: ";
    y->Print();
    x0->Grad()->Print();
    x1->Grad()->Print();
    x2->Grad()->Print();
}

Variable compositeFunction(Variable& input) {
    auto x = make_shared<Variable>(input);

    return *square(exp(square(x)));
}
void numericalDiffExample() {
    vector<int> shape = {1};
    shared_ptr<vector<float>> data = make_shared<vector<float>>(vector<float>{0.5});

    auto t = make_shared<Tensor>(shape, data);
    Variable x(t, "t");

    auto dy = numerical_diff(compositeFunction, x);

    dy->Print();
}

shared_ptr<Variable> sphere(const shared_ptr<Variable>& x, const shared_ptr<Variable>& y) {
    auto z = (x ^ 2) + (y ^ 2);

    return z;
}
shared_ptr<Variable> matyas(const shared_ptr<Variable>& x, const shared_ptr<Variable>& y) {
    auto z = 0.26 * ((x ^ 2) + (y ^ 2)) - 0.48 * x * y;

    return z;
}
shared_ptr<Variable> goldstein(const shared_ptr<Variable>& x, const shared_ptr<Variable>& y) {
    auto z = (1+((x+y+1)^2)*(19-14*x+3*(x^2)-14*y+6*x*y+3*(y^2)))
                                * (30+((2*x-3*y)^2)*(18-32*x+12*(x^2)+48*y-36*x*y+27*(y^2)));

    return z;
}

void sphereExample() {
    auto x = Variable::New(vector<int>{1}, vector<float>{1.0}, "x");
    auto y = Variable::New(vector<int>{1}, vector<float>{1.0}, "y");

    auto z = goldstein(x,y);//matyas(x, y);//sphere(x, y);

    z->backward();

    x->Grad()->Print();
    y->Grad()->Print();

    z->SetName("z");

    DrawComputationGraph(z, "1.png");
}
int factorial(int n) {
    if (n == 0 || n == 1) return 1;

    int c = 1;
    for (int i = 2; i <= n; i++) {
        c *= i;
    }

    return c;
}
shared_ptr<Variable> mySin(const shared_ptr<Variable>& x, float threshold=0.0001) {
    auto y = Variable::New(vector<int>{1}, vector<float>{0});

    for (int i = 0; i < 100000; i++) {
        auto c = std::powf(-1, float(i)) / float(factorial(2 * i + 1));

        auto t = c * (x ^ float(2 * i + 1));
        y = y + t;

        cout << "i:" << i << endl;
        if (std::abs(t->Data()->DataAt(vector<int>{0})) < threshold) {
            break;
        }
    }

    return y;
}
void sinExample() {
    auto x = Variable::New(vector<int>{1}, vector<float>{1}, "x");
    auto y = sin(x);

    y->backward();

    for (int i = 0; i < 3; i++) {
        auto gx = x->Grad();
        x->ClearGrad();
        gx->backward();

        x->Grad()->Print();
    }

    //y->Print();
    //x->Grad()->Print();
    //y->SetName("y");
    //DrawComputationGraph(y, "1.png");
}

shared_ptr<Variable> rosenBlock(const shared_ptr<Variable>& x, const shared_ptr<Variable>& y) {
    auto z = 100 * ((y - (x ^ 2)) ^ 2) + ((x - 1) ^ 2);

    return z;
}

void rosenBlockExample() {
    auto x = Variable::New(vector<int>{1}, vector<float>{0}, "x");
    auto y = Variable::New(vector<int>{1}, vector<float>{2}, "y");

    float lr = 0.001;
    int iters = 10000;

    for (int i = 0; i < iters; i++) {
        cout << "----" << endl;
        x->Print();
        y->Print();

        auto z = rosenBlock(x, y);
        //z->Print();

        x->ClearGrad();
        y->ClearGrad();

        z->backward();

        //x->Grad()->Print();
        //y->Grad()->Print();

        x->Data() -= lr * x->Grad()->Data();
        y->Data() -= lr * y->Grad()->Data();
    }
}

void tanhExample() {
    auto x = Variable::New(vector<int>{1}, vector<float>{1}, "x");
    auto y = tanh(x);

    y->backward();

    for (int i = 0; i < 3; i++) {
        auto gx = x->Grad();
        x->ClearGrad();
        gx->backward();

        //x->Grad()->Print();
    }

    //y->Print();
    auto gx = x->Grad();
    gx->SetName("gx");
    gx->Print();
    //y->SetName("y");
    DrawComputationGraph(gx, "1.png");
}

void sumExample() {
    auto a = Variable::New(vector<int>{2,2,3}, vector<float>{0,1,2,3,4,5,6,7,8,9,10,11}, "x");
    cout << "---a---" << endl;
    a->Print();
    a->PrintShape();
    auto b = sum(a, vector<int>{0,1});
    cout << "---b---" << endl;
    b->Print();
    b->PrintShape();
    auto c = sum(a, vector<int>{1,2});
    cout << "---c---" << endl;
    c->Print();
    c->PrintShape();
    auto d = sum(a, vector<int>{0,1,2});
    cout << "---d---" << endl;
    d->Print();
    d->PrintShape();
    auto e = sum(a, vector<int>{0});
    cout << "---e---" << endl;
    e->Print();
    e->PrintShape();
    auto f = sum(a, vector<int>{1});
    cout << "---f---" << endl;
    f->Print();
    f->PrintShape();
    auto g = sum(a, vector<int>{2});
    cout << "---g---" << endl;
    g->Print();
    g->PrintShape();
}

void sumBackwardExample() {
    auto x = Variable::New(vector<int>{2,3}, vector<float>{1,2,3,4,5,6}, "x");
    //cout << "---a---" << endl;
    //x->Print();
    //x->Data()->PrintShape();
    auto y = sum(x, vector<int>{0});
    y->backward();
    //cout << "---b---" << endl;
    y->Print();
    x->Grad()->Print();
}

void sumToExample() {
    auto a = Variable::New(vector<int>{2,3}, vector<float>{1,2,3,4,5,6}, "x");
    cout << "---a---" << endl;
    a->Print();
    a->Data()->PrintShape();
    auto b = sumTo(a, vector<int>{1,3});
    cout << "---b---" << endl;
    b->Print();
    b->Data()->PrintShape();
    auto c = sumTo(a, vector<int>{2,1});
    cout << "---c---" << endl;
    c->Print();
    c->Data()->PrintShape();
}

void broadcastToExample() {
    auto a = Variable::New(vector<int>{2,3}, vector<float>{1,2,3,4,5,6}, "x");
    cout << "---a---" << endl;
    a->Print();
    auto b = transpose(a, vector<int>{1,0});
    cout << "---b---" << endl;
    b->Print();
    auto c = broadcastTo(b, vector<int>{2,3,2});
    cout << "---c---" << endl;
    c->Print();

    auto d = Variable::New(vector<int>{1}, vector<float>{6}, "x");
    cout << "---d---" << endl;
    d->Print();
    auto e = broadcastTo(d, vector<int>{3});
    cout << "---e---" << endl;
    e->Print();

    d = Variable::New(vector<int>{1}, vector<float>{6}, "x");
    cout << "---d---" << endl;
    d->Print();
    e = broadcastTo(d, vector<int>{3,3});
    cout << "---e---" << endl;
    e->Print();

    a = Variable::New(vector<int>{2,1,1,3}, vector<float>{1,2,3,4,5,6}, "x");
    cout << "---a---" << endl;
    a->Print();
    c = broadcastTo(a, vector<int>{2,2,2,3});
    cout << "---c---" << endl;
    c->Print();
}

void matMulExample() {
    auto a = Variable::New(vector<int>{2,3}, vector<float>{1,2,3,4,5,6}, "x");
    auto b = Variable::New(vector<int>{3,4}, vector<float>{0,1,2,3,4,5,6,7,8,9,10,11}, "x");

    auto c = matMul(a, b);
    c->PrintShape();
    c->Print();

    c->backward();
    a->Grad()->PrintShape();
    b->Grad()->PrintShape();
}

shared_ptr<Variable> linearRegressionPredict(const shared_ptr<Variable>& x, const shared_ptr<Variable>& W, const shared_ptr<Variable>& b) {
    auto y = linear(x, W, b); //matMul(x, W) + b;
    return y;
}

void linearRegressionExample() {
    auto data = UniformRandomData(100);
    auto noiseData = UniformRandomData(100);

    auto x = Variable::New(vector<int>{100,1}, data, "x");
    auto noise = Variable::New(vector<int>{100,1}, noiseData, "noise");
    auto y = 5 + 2 * x + noise;

    auto W = Variable::New(vector<int>{1,1}, vector<float>{0}, "W");
    auto b = Variable::New(vector<int>{1}, vector<float>{0}, "b");

    float lr = 0.1;
    int iters = 1000;

    for (int i = 0; i < iters; i++) {
        auto y_predict = linearRegressionPredict(x, W, b);
        auto loss = meanSquaredError(y, y_predict);

        W->ClearGrad();
        b->ClearGrad();

        loss->backward();

        //x->Grad()->Print();
        //y->Grad()->Print();

        W->Data() -= lr * W->Grad()->Data();
        b->Data() -= lr * b->Grad()->Data();

        W->Print();
        b->Print();
        loss->Print();
    }
}

shared_ptr<Variable> sinRegressionPredict(const shared_ptr<Variable>& x, Linear& l1, Linear& l2) {
    auto y = l1(x); //l1.Forward(x)[0];
    y = sigmoid(y);
    y = l2(y);

    return y;
}

void sinRegressionExample() {
    auto data = UniformRandomData(100);
    auto noiseData = UniformRandomData(100);

    auto x = Variable::New(vector<int>{100,1}, data, "x");
    auto noise = Variable::New(vector<int>{100,1}, noiseData, "noise");
    auto y = sin(2 * PI * x) + noise;

    int I = 1, H = 10, O = 1;

    auto model = MLP({I, H, O});
    auto optimizer = SGD(model.Parameters(), 0.2);

    int iters = 10000;

    for (int i = 0; i < iters; ++i) {
        auto y_predict = model(x);//sinRegressionPredict(x, l1, l2);
        auto loss = meanSquaredError(y, y_predict);
        loss->SetName("Loss");

        optimizer.ClearGrads();

        loss->backward();

        optimizer.Update();

        if (i % 1000 == 0 || i == iters - 1) {
            cout << "Loss:";
            loss->Print();
        }
    }

    cout << "Training done. Predict:" << endl;
    auto a = Variable::New(vector<int>{11,1}, vector<float>{0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1}, "x");
    auto b = model(a);//sinRegressionPredict(a, l1, l2);
    b->Print();
}

int main(int argc, char *argv[]) {
    //addExample();
    //VariableAndFloatExample();
    //sphereExample();
    //sinExample();
    //rosenBlockExample();
    tanhExample();
    //reshapeExample();
    //transposeExample();
    //sumBackwardExample();
    //broadcastToExample();
    //matMulExample();
    //linearRegressionExample();
    //sinRegressionExample();

    return 0;
}