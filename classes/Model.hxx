//
// Created by malav on 4/26/2022.
//

#ifndef ANN_MODEL_HXX
#define ANN_MODEL_HXX

#include <vector>
#include "layers/layer_types.hxx"
#include "helpers/visitors.hxx"

class Model {
    public:

        // Create a Model object
        Model() = default;

        // Copy constructor
        Model(const Model&);

        // Destructor to release allocated memory
        ~Model() = default;

        // Add a layer to the model
        template<typename LayerType, typename... Args>
        void Add(Args... args) {network.push_back(new LayerType(args...));}

        // train the network on the _data

        void /*return_type TBD */ Train(size_t opt /* args to be filled */ );

        // return outshape of a layer
        Dims get_outshape(size_t idx){ return boost::apply_visitor(Outshape_visitor(), network[idx]);}

        // make a forward pass through the network
        void Forward(Vector<double>& input, Vector<double>& output);

        // make a Backward pass through the network
        void Backward(Vector<double>& dLdY, Vector<double>& dLdX);

        // make a pass through the network, updating all the parameters
        template<typename Optimizer>
        void Update_Params(Optimizer* optimizer, size_t normalizer);

        // get number of layers in model
        size_t get_size() const {return network.size();}

        // get const reference to vector of layers
        std::vector<LayerTypes> get_network() const {return network;}

    private:
        std::vector<LayerTypes> network;

};

#include "Model_impl.hxx"
#endif //ANN_MODEL_HXX
