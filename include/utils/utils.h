//
// Created by Fangbo Zhang on 2023/6/17.
//

#ifndef TINYLEARNING_UTILS_H
#define TINYLEARNING_UTILS_H

#include <vector>

using namespace std;

namespace TinyLearning {
    vector<float> UniformRandomData(int n);
    vector<float> NormalRandomData(int n);
    vector<float> ScaledNormalRandomData(int n, float scale);
}

#endif //TINYLEARNING_UTILS_H
