//
// Created by malav on 6/23/2022.
//

#ifndef ANN_MODEL_IMPL_HXX
#define ANN_MODEL_IMPL_HXX

#include "Model.hxx"
#include <algorithm>
#include <iostream>
#include "optimizers/optimizers.hxx"

void
Model::Forward(Vector<double> &input, Vector<double>& output)
{
    Forward_visitor visitor{};
    visitor.input = &input;


    for (LayerTypes layer : network)
    {
        Dims out_shape = boost::apply_visitor(Outshape_visitor(), layer);

        visitor.output = new Vector<double>(out_shape.width*out_shape.height);
        boost::apply_visitor(visitor, layer);

        delete visitor.input;
        visitor.input = visitor.output;
    }

    output = *(visitor.output);
}

void Model::Backward(Vector<double> &dLdY, Vector<double>& dLdX)
{
    Backward_visitor visitor{};
    visitor.dLdY = &dLdY;

    // i must be type int or else code fails! i-- turn i = 0 into i = largest unsigned int possible
    for (int i = network.size() - 1; i >= 0; i--)
    {
        LayerTypes layer = network[i];

        Dims in_shape =  boost::apply_visitor(Inshape_visitor(), layer);

        visitor.dLdX = new Vector<double>(in_shape.width * in_shape.height);
        boost::apply_visitor(visitor, layer);

        delete visitor.dLdY;
        visitor.dLdY = visitor.dLdX;
    }

    dLdX = *(visitor.dLdX);
}

template<typename Optimizer>
void Model::Update_Params(Optimizer* optimizer, size_t normalizer)
{
    // create visitor object
    Update_parameters_visitor<Optimizer> visitor {};

    // give visitor the optimizer and normalizer values
    visitor.normalizer = normalizer;
    visitor.optimizer = optimizer;

    // send visitor to each layer to update weights and biases
    for (LayerTypes layer : network)
    {
        boost::apply_visitor(visitor, layer);
    }
}

void /* return type TBD */ Model::Train(size_t opt /* args TBD */)
{
    // initialize the optimizer using a switch statement

    // determine if number of training points is divisible by the batch_size
    //      - if there is no remainder, we will be updating the parameters (num training points) / (batch_size) times
    //      - if there is a remainder, we will update the parameters |_ (num training points) / (batch_size) _| times
    //        and then proceed to train on the remainder of the training set ( num training points % batch_size)


    // for each data point in my training set:
    //      - make a Forward pass
    //      - compute dLdY, the loss gradient at the output layer as a result of this Forward pass
    //      - make a Backward pass, backpropagating the calculated loss gradient
    //      - if we have sent a batch_size amount of data points forward and backward after this last pass, update the
    //        parameters in the networks using Update_Params function
}

#endif //ANN_MODEL_IMPL_HXX
