cmake_minimum_required(VERSION 3.25)
project(TinyLearning)

set(CMAKE_CXX_STANDARD 11)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_subdirectory(src)
add_subdirectory(example)

#add_library(TinyLearning src/core/tensor/tensor.cpp include/core/tensor/tensor.h example/main.cpp src/core/function.cpp include/core/function.h src/core/variable.cpp include/core/variable.h src/core/functions/square.cpp include/core/functions/square.h src/core/functions/exp.cpp include/core/functions/exp.h src/core/numerical_diff.cpp include/core/numerical_diff.h include/core/functions.h src/core/functions.cpp include/core/functions/add.h src/core/functions/add.cpp include/core/functions/mul.h src/core/functions/mul.cpp include/core/functions/neg.h src/core/functions/neg.cpp include/core/functions/sub.h src/core/functions/sub.cpp include/core/functions/div.h src/core/functions/div.cpp include/core/functions/pow.h src/core/functions/pow.cpp include/graphviz/graphviz.h src/graphviz/graphviz.cpp include/core/functions/sin.h src/core/functions/sin.cpp include/core/functions/cos.h src/core/functions/cos.cpp include/core/functions/tanh.h src/core/functions/tanh.cpp include/core/functions/reshape.h src/core/functions/reshape.cpp include/core/functions/transpose.h src/core/functions/transpose.cpp include/core/functions/sum.h src/core/functions/sum.cpp include/core/functions/broadcastTo.h src/core/functions/broadcastTo.cpp include/core/functions/sumTo.h src/core/functions/sumTo.cpp include/core/functions/matMul.h src/core/functions/matMul.cpp include/core/functions/meanSquaredError.h src/core/functions/meanSquaredError.cpp include/core/functions/sigmoid.h src/core/functions/sigmoid.cpp include/core/functions/linear.h src/core/functions/linear.cpp include/core/module.h src/core/module.cpp include/core/layers/linear.h src/core/layers/linear.cpp include/utils/utils.h src/utils/utils.cpp include/core/models/mlp.h src/core/models/mlp.cpp include/core/optimizer.h src/core/optimizer.cpp include/core/optimizers/sgd.h src/core/optimizers/sgd.cpp)
