cmake_minimum_required(VERSION 3.25)
project(TinyLearningLib)

set(CMAKE_CXX_STANDARD 11)

add_library(TinyLearning core/tensor/tensor.cpp core/function.cpp core/variable.cpp core/functions/square.cpp core/functions/exp.cpp core/numerical_diff.cpp core/functions.cpp core/functions/add.cpp core/functions/mul.cpp core/functions/neg.cpp core/functions/sub.cpp core/functions/div.cpp core/functions/pow.cpp graphviz/graphviz.cpp core/functions/sin.cpp core/functions/cos.cpp core/functions/tanh.cpp core/functions/reshape.cpp core/functions/transpose.cpp core/functions/sum.cpp core/functions/broadcastTo.cpp core/functions/sumTo.cpp core/functions/matMul.cpp core/functions/meanSquaredError.cpp core/functions/sigmoid.cpp core/functions/linear.cpp core/module.cpp core/layers/linear.cpp utils/utils.cpp core/models/mlp.cpp core/optimizer.cpp core/optimizers/sgd.cpp)

add_library(sub::TinyLearning ALIAS TinyLearning)