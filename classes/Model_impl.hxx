//
// Created by malav on 6/23/2022.
//

#ifndef ANN_MODEL_IMPL_HXX
#define ANN_MODEL_IMPL_HXX

#include "Model.hxx"
#include <algorithm>
#include <iostream>

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
void Model::Update_Params(Optimizer optimizer, size_t normalizer)
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

#endif //ANN_MODEL_IMPL_HXX
