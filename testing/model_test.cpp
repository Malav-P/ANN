//
// Created by malav on 6/23/2022.
//

#include "../classes/Model.hxx"
#include "../classes/loss functions/loss_functions.hxx"

int main()
{
    Model<CrossEntropy> model;

    double in_arr[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9,
                         10, 11, 12, 13, 14, 15, 16};

    Vector<double>* input_ptr = new Vector<double>(16, in_arr);

    double fltr_arr[4] = {1,1,1,1};
    Mat<double> fltr(2, fltr_arr);

    double out_arr[2] = {4, 2};
    Vector<double> output;


    Vector<double> dLdY(2, out_arr);
    Vector<double> dLdX;

    model.Add<Convolution>(  4    // input width
                           , 4    // input height
                           , fltr // filter
                           , 1    // horizontal stride length
                           , 1    // vertical stride length
                           , true
                           );

    model.Add<MaxPool>(  model.get_outshape(0).width   // input width = 3
                       , model.get_outshape(0).height  // input height = 3
                       , 2  // filter width
                       , 2  // filter height
                       , 1  // horizontal stride length
                       , 1  // vertical stride length
                       );


    model.Add<Linear<Tanh>>(  model.get_outshape(1).width * model.get_outshape(1).height  // input size
                            , 2                                                                   // output size
                            );

    model.Add<Softmax>(model.get_outshape(2).width * model.get_outshape(2).height // input size
                        );



    model.Forward(*input_ptr, output);

    // must call model.Forward(input, output) before calling Backward function because
    // _local_input member variable must be filled
    double tar_arr[2] = {1, 0};
    Vector<double> target(output.get_len(), tar_arr);
    dLdY = model.get_grad(output, target);

    model.Backward(dLdY, dLdX);

    // choose an optimizer
    Momentum optimizer(model, 0.1, 0.9);
//    SGD optimizer;

    // update parameters in the network
    model.Update_Params(&optimizer,1);

}

