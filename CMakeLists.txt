cmake_minimum_required(VERSION 3.21)
project(ANN)

set(CMAKE_CXX_STANDARD 14)

include_directories(../../../../../MinGW/include/boost_1_79_0)

add_executable(linalg_test testing/linalg_test.cpp)
add_executable(convolution_test testing/convolution_test.cpp)
add_executable(maxpool_test testing/maxpool_test.cpp)
add_executable(meanpool_test testing/meanpool_test.cpp)
add_executable(linear_test testing/linear_test.cpp)
add_executable(model_test testing/model_test.cpp)

