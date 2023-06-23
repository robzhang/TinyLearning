//
// Created by Fangbo Zhang on 2023/6/17.
//
#include <vector>
#include <random>

using namespace std;

namespace TinyLearning {
    template<typename T>
    vector<float> randomData(int n) {
        static std::random_device randomDevice;
        static default_random_engine e(randomDevice());

        T u(0.0, 1.0);

        vector<float> data(n);
        for (auto i = 0; i < n; ++i) {
            data[i] = u(e);
        }

        return data;
    }

    vector<float> UniformRandomData(int n) {
        return randomData<uniform_real_distribution<float>>(n);
    }

    vector<float> NormalRandomData(int n) {
        return randomData<normal_distribution<float>>(n);
    }

    vector<float> ScaledNormalRandomData(int n, float scale) {
        auto data = NormalRandomData(n);

        for (auto &d: data) {
            d *= scale;
        }

        return data;
    }
}