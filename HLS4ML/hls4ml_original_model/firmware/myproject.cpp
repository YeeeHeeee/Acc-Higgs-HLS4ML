#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    input_t x[N_INPUT_1_1],
    result_t layer9_out[N_LAYER_8]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=x complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=x,layer9_out 
    #pragma HLS PIPELINE 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<model_default_t, 8960>(w2, "w2.txt");
        nnet::load_weights_from_txt<model_default_t, 64>(b2, "b2.txt");
        nnet::load_weights_from_txt<model_default_t, 2048>(w4, "w4.txt");
        nnet::load_weights_from_txt<model_default_t, 32>(b4, "b4.txt");
        nnet::load_weights_from_txt<model_default_t, 512>(w6, "w6.txt");
        nnet::load_weights_from_txt<model_default_t, 16>(b6, "b6.txt");
        nnet::load_weights_from_txt<model_default_t, 16>(w8, "w8.txt");
        nnet::load_weights_from_txt<model_default_t, 1>(b8, "b8.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    layer2_t layer2_out[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    nnet::dense<input_t, layer2_t, config2>(x, layer2_out, w2, b2); // layer0

    layer3_t layer3_out[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
    nnet::relu<layer2_t, layer3_t, ReLU_config3>(layer2_out, layer3_out); // relu

    layer4_t layer4_out[N_LAYER_4];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    nnet::dense<layer3_t, layer4_t, config4>(layer3_out, layer4_out, w4, b4); // layer1

    layer5_t layer5_out[N_LAYER_4];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    nnet::relu<layer4_t, layer5_t, ReLU_config5>(layer4_out, layer5_out); // relu_1

    layer6_t layer6_out[N_LAYER_6];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
    nnet::dense<layer5_t, layer6_t, config6>(layer5_out, layer6_out, w6, b6); // layer2

    layer7_t layer7_out[N_LAYER_6];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    nnet::relu<layer6_t, layer7_t, ReLU_config7>(layer6_out, layer7_out); // relu_2

    layer8_t layer8_out[N_LAYER_8];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0
    nnet::dense<layer7_t, layer8_t, config8>(layer7_out, layer8_out, w8, b8); // layer3

    nnet::sigmoid<layer8_t, result_t, Sigmoid_config9>(layer8_out, layer9_out); // sigmoid

}
