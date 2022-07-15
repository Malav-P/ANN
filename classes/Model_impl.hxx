//
// Created by malav on 6/23/2022.
//

#ifndef ANN_MODEL_IMPL_HXX
#define ANN_MODEL_IMPL_HXX

#include "Model.hxx"
#include <algorithm>
#include <iostream>
#include "optimizers/optimizers.hxx"

template<typename LossFunction>
void Model<LossFunction>::Forward(Vector<double> &input, Vector<double>& output)
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

template<typename LossFunction>
void Model<LossFunction>::Backward(Vector<double> &dLdY, Vector<double>& dLdX)
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

template<typename LossFunction>
template<typename Optimizer>
void Model<LossFunction>::Update_Params(Optimizer* optimizer, size_t normalizer)
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

    // reset the optimizer for the next pass through the network
    (*optimizer).reset();
}

template<typename LossFunction>
void Model<LossFunction>::Train(size_t opt, DataSet& training_set /* args TBD */)
{

    // initialize the optimizer using a switch statement

    void* optimizer;

    switch (opt)
    {
        case 0: // SGD
        {
            optimizer = static_cast<SGD*> (new SGD());
            break;
        }

        case 1: // Momentum
        {
            optimizer = static_cast<Momentum*>(new Momentum(*this, 0.1, 0.9)) ;
            break;
        }

        default : // default case (this needs to be changed)
        {
            optimizer = static_cast<int*>(new int);
        }
    }


    // determine if number of training points is divisible by the batch_size
    //      - if there is no remainder, we will be updating the parameters (num training points) / (batch_size) times
    //      - if there is a remainder, we will update the parameters |_ (num training points) / (batch_size) _| times
    //        and then proceed to train on the remainder of the training set ( num training points % batch_size)

    size_t num_training_points, batch_size, remainder;

    num_training_points = training_set.shape.width;

    remainder = num_training_points % batch_size;

    // for each data point in my training set:
    //      - make a Forward pass
    //      - compute dLdY, the loss gradient at the output layer as a result of this Forward pass
    //      - make a Backward pass, backpropagating the calculated loss gradient
    //      - if we have sent a batch_size amount of data points forward and backward after this last pass, update the
    //        parameters in the networks using Update_Params function

    Vector<double> output, dLdY, dLdX;
    size_t count = 0;

    for (Vector_Pair datapoint : training_set.datapoints)
    {
        // make forward pass, datapoint.first is the input
        Forward((datapoint.first), output);

        // compute dLdY, datapoint.second is the label
        dLdY = loss.grad(output, (datapoint.second));

        // make a backward pass
        Backward(dLdY, dLdX);
        count += 1;

        // update parameters if we have propagated batch_size number of samples
        if (count % batch_size == 0) { Update_Params(optimizer, batch_size); }
    }

    // if remainder exists we can update the model with the remaining datapoints
    if (remainder != 0) { Update_Params(optimizer, remainder);}

    // free memory for optimizer
    delete optimizer;
}

#endif //ANN_MODEL_IMPL_HXX
