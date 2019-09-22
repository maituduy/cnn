#include <iostream>
#include <armadillo>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <time.h>
#include <json.hpp>

#include "nn_ops.h"
#include "f.h"

#include "layers/input.h"
#include "layers/conv2d.h"
#include "layers/pool2d.h"
#include "layers/conv2d_transpose.h"
#include "layers/batch_norm.h"

#include "parser.h"

#include "model.h"

using json = nlohmann::json;
using namespace arma;
using namespace cv;
using namespace layer;
using namespace parser;

class Tictoc {
    public:
        Tictoc(){};

        template<typename T>
        void start(T f, std::string title) {
            clock_t tStart = clock();
            f();
            printf("Time taken [%s]: %.2fs\n", title.c_str(), (double)(clock() - tStart)/CLOCKS_PER_SEC);
        };
};

int main() {
    Shape input_shape(10, 16,16,3);
    
    Model *model = new Model();
    model
        ->add(layer::Input(input_shape))
        ->add(layer::Conv2d(1, 3, Padding::SAME, 2, mtype::Activation::RELU))
        ->add(layer::Conv2d(4, 3, Padding::SAME, 1, mtype::Activation::RELU))
        ->add(layer::Conv2dTranspose(1, 5, Padding::SAME, 2, mtype::Activation::RELU))
        ->add(layer::Pooling2d(2))
        ->add(layer::Conv2d (32, 3, Padding::VALID, 1, mtype::Activation::SIGMOID))
        ->add(layer::Pooling2d(2, PoolingMode::AVERAGE_TF, Padding::VALID, 2))
        ->add(layer::BatchNormalization())
        ->add(layer::Conv2dTranspose(1, 3, Padding::VALID, 1, mtype::Activation::SIGMOID))
        ->add(layer::Conv2d(32, 3, Padding::VALID, 1, mtype::Activation::SIGMOID))
        ->add(layer::BatchNormalization())
        ->separate()
    ;
    // model->summary();
    // model->load_weights("/home/mxw/dev/notebook/weights.dat");
    auto in = model->get_input("/home/mxw/dev/notebook/input.dat");
    auto output = model->predict(in);
    output.print();
    
    
    return 0;
}
