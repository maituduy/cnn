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


int main() {
    Tictoc tictoc;

    Shape input_shape(10, 16,16,3);
    layer::Input input(input_shape);
    auto c1 = (*new layer::Conv2d(1, 3, Padding::SAME, 2, mtype::Activation::RELU))(&input);
    auto c2 = (*new layer::Conv2d(4, 3, Padding::SAME, 1, mtype::Activation::RELU))(c1);
    auto c3 = (*new layer::Conv2dTranspose(1, 5, Padding::SAME, 2, mtype::Activation::RELU))(c2);
    auto c4 = (*new layer::Pooling2d(2))(c3);
    auto c5 = (*new layer::Conv2d (32, 3, Padding::VALID, 1, mtype::Activation::SIGMOID))(c4);
    auto c6 = (*new layer::Pooling2d(2, PoolingMode::AVERAGE_TF, Padding::VALID, 2))(c5);
    auto c7 = (*new layer::BatchNormalization())(c6);
    auto c8 = (*new layer::Conv2dTranspose(1, 3, Padding::VALID, 1, mtype::Activation::SIGMOID))(c7);
    auto c9 = (*new layer::Conv2d(32, 3, Padding::VALID, 1, mtype::Activation::SIGMOID))(c8);
    auto c10 = (*new layer::BatchNormalization())(c9);
    
    tictoc.start([&]() {
        Model model(c10);
        // model.summary();
        model.load_weights("/home/mxw/dev/notebook/weights.dat");
        auto in = model.get_input("/home/mxw/dev/notebook/input.dat");
        auto output = model.predict(in);
        output(0).slice(31).print();
        
    }, "test");
    
    return 0;
}
