//
// Created by malav on 7/15/2022.
//

#ifndef ANN_CROSS_ENTROPY_HXX
#define ANN_CROSS_ENTROPY_HXX

#include "../lin_alg/data_types.hxx"
#include "../prereqs.hxx"

class CrossEntropy {

    public:

        double loss(Vector<double>& output, Vector<double>& target)
        {
            // assert that number of elements in other vector and this vector are equal
            assert(output.get_len() == target.get_len());

            // initialize return variable
            double loss = 0;

            // compute loss
            for (size_t i = 0; i < output.get_len(); i++) { loss -= target[i] * log(output[i]); }

            // return result
            return loss;
        }

        Vector<double> grad(Vector<double>& output, Vector<double>& target) {return output.edivide(target);}

    private:

};

#endif //ANN_CROSS_ENTROPY_HXX
