//
// Created by malav on 7/22/2022.
//

#include "../mnist/read_mnist.hxx"
#include <iostream>

int main()
{
    DataSet container = read_mnist(10);

//    for (size_t i = 0; i<10; i++)
//    {
//        std::cout << container.datapoints[9].second[i]<< "\n";
//    }

    return 0;
}