//
// Created by Fangbo Zhang on 2023/6/19.
//
#include <iostream>

#include "core/models/mlp.h"
#include "core/layers/linear.h"

using namespace std;

namespace TinyLearning {
    MLP::MLP(const std::initializer_list<int> &sizes, const function<shared_ptr<Variable>(const shared_ptr<Variable>&)>& activation) : activation_(activation) {
        this->layers_.reserve(sizes.size() - 1);

        auto I = *sizes.begin();
        for (auto iter = sizes.begin() + 1; iter != sizes.end(); iter++) {
            auto O = *iter;
            cout << "layer parameter size:" << I << " x " << O << endl;
            auto layer = make_shared<Linear>(I, O);
            this->layers_.push_back(layer);

            I = O;
        }
    }

    shared_ptr<Variable> MLP::Forward(const shared_ptr<Variable>& x) {
        auto layer = this->layers_[0];

        shared_ptr<Variable> y = (*layer)(x);

        for (auto i = 1; i < this->layers_.size(); i++) {
            y = this->activation_(y);

            layer = this->layers_[i];

            y = (*layer)(y);
        }

        return y;
    }

    vector<shared_ptr<Variable>> MLP::Parameters() {
        if (this->parameters_.empty()) {
            this->setupParameters();
        }

        return this->parameters_;
    }

    void MLP::setupParameters() {
        for (const auto& layer : this->layers_) {
            auto layerParameters = layer->Parameters();
            this->parameters_.insert(this->parameters_.end(), layerParameters.begin(), layerParameters.end());
        }
    }
}