#include <iostream>
#include <time.h>
#include <json.hpp>
#include "armadillo"
#include "parser.h"
#include "u4.h"

using json = nlohmann::json;
using namespace arma;
using namespace parser;
using namespace model_zoo;

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
    Shape input_shape(5, 32,32,1);
    auto u4 = new U4(input_shape, 2);
    auto model = u4->get();

    model->load_weights("../data/weights.dat");
    auto in = Parser::get_input("../data/input.dat");

    auto output = model->predict(in);
    output(0).slice(0).print();

//    example
//    Shape input_shape(10, 16,16,3);
//    layer::Input ip(input_shape);
//    auto *model = new Model();
//    model
//        ->add(layer::Input(input_shape))
//        ->add(layer::Conv2d(1, 3, Padding::SAME, 2, Func::RELU))
//        ->add(layer::Conv2d(4, 3, Padding::SAME, 1, Func::RELU))
//        ->add(layer::Conv2dTranspose(1, 5, Padding::SAME, 2, Func::NONE))
//        ->add(layer::Activation(Func::RELU))
//        ->add(layer::Pooling2d(2))
//        ->add(layer::Conv2d (32, 3, Padding::VALID, 1, Func::SIGMOID))
//        ->add(layer::Pooling2d(2, PoolingMode::AVERAGE_TF, Padding::VALID, 2))
//        ->add(layer::BatchNormalization())
//        ->add(layer::Conv2dTranspose(1, 3, Padding::VALID, 1, Func::SIGMOID))
//        ->add(layer::Conv2d(32, 3, Padding::VALID, 1, Func::SIGMOID))
//        ->add(layer::BatchNormalization())
//        ->separate()
//    ;
//    model->summary();
//    model->load_weights("../data/weights.dat");
//    auto in = model->get_input("../data/input.dat");
//
//    auto output = model->predict(in);
//    output(0).slice(0).print();

    return 0;
}
