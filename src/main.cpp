#include <iostream>
#include <armadillo>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <variant>
#include <string>
#include <time.h>
#include <json.hpp>
#include <map>

#include "nn_ops.h"
#include "f.h"
#include "layer.h"
#include "layers/input.cpp"
#include "layers/conv2d.cpp"
#include "layers/pool2d.cpp"
#include "layers/conv2d_transpose.cpp"
#include "layers/batch_norm.cpp"

#include "parser.h"

#include "model.h"

using json = nlohmann::json;
using namespace arma;
using namespace ops;
using namespace f;
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

class A {
    int a;
    public:
        A() {
            std::cout << 111;
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

    model->separate();
    model->load_weights("/home/mxw/dev/notebook/weights.dat");
    auto in = model->get_input("/home/mxw/dev/notebook/input.dat");
    std::cout << "??";
    auto output = model->predict(in);
    output(0).slice(31).print();
    
    
    return 0;
}
