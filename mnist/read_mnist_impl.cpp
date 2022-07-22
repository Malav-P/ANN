//
// Created by malav on 7/18/2022.
//

// this file reads in the training data and testing data into formats acceptable by the CNN architecture

#include "read_mnist.hxx"
#include <iostream>


DataSet read_mnist(size_t N_TRAIN)
{

    DataSet container({N_TRAIN, 784});

    FILE *fp;

    int row, col, label;

    char buf[10000];
    double label_buf[10]{0};
    double vector_buf[784];

    char* tok;

    fp = fopen("C:/Users/malav/CLionProjects/ANN-master/ANN-master/mnist/mnist_train.csv","r");
    if(fp == nullptr) {
        perror("Error opening file");
        exit(1);
    }

    for(row=0;row < N_TRAIN;row++){

        col = 0;

        fgets(buf, sizeof(buf), fp);

        tok = strtok(buf, ",");
        label = atoi(tok);

        // fill in buffer with correct truth value
        label_buf[label] = 1.0;

        tok = strtok(nullptr, ",");
        while (tok != nullptr){
            vector_buf[col] = strtod(tok, nullptr)/255.0;
            col +=1;
            tok = strtok(nullptr, ",");
        }


        container.datapoints.emplace_back(Vector<double>(784, vector_buf), Vector<double>(10, label_buf));

        // return buffer to original state
        label_buf[label] = 0;
    }

    fclose(fp);

    return container;

}


