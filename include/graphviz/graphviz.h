//
// Created by Fangbo Zhang on 2023/6/8.
//

#ifndef TINYLEARNING_GRAPHVIZ_H
#define TINYLEARNING_GRAPHVIZ_H

#include "../core/variable.h"

namespace TinyLearning {
    string DrawGraphviz(const shared_ptr<Variable>& v);
    void DrawComputationGraph(const shared_ptr<Variable>& v, const string& filename);
}

#endif //TINYLEARNING_GRAPHVIZ_H
