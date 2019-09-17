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
#include "activation.h"
#include "f.h"
#include "layer.h"
#include "layers/input.cpp"
#include "layers/conv2d.cpp"

#include "model.h"

using json = nlohmann::json;
using namespace arma;
using namespace ops;
using namespace f;
using namespace cv;
using namespace layer;
// template<class T3>
// arma::Mat <T3> cvMat2armaMat(cv::Mat & cvMatIn) 
// { 
//     return arma::Mat <T3> (cvMatIn.data, cvMatIn.rows, cvMatIn.cols); 
// }

int main() {    
    // arma::mat a = {
    //      {2,3.2,6},
    //      {1,5,-2},
    //      {5,1,7}
    //  };

    // arma::mat kernel = {
    //     {1,-1,2}, 
    //     {1,1,-2},
    //     {1,5,6},   
    // };

    // arma::mat c(arma::linspace(-255,0,256));
    // arma::mat d = arma::reshape(c, 16, 16).t();

    // cv::Mat image;
    // image = cv::imread("/home/xmw//Desktop/1.png", CV_LOAD_IMAGE_COLOR);
    
    // cv::Mat img(2, 2, 2, CV_32FC1);
    // cv::randu(img, cv::Scalar(0), cv::Scalar(255));
        
    // std::cout << "img Test" << img << std::endl;
    // cv::Mat imgt(img.t());
        
    // // mat to armadillo
    // arma::fmat armaConv(imgt.ptr<float>(), 2, 2);
    // std::cout << "armaConv" << std::endl << armaConv << std::endl;
    // config["e"] = 1;
    // config["asd"] = Shape(1,2,3,1);

    arma::field<cube> field(8);
    for (size_t i = 0; i < field.n_elem; i++)
    {
        field(i) = arma::randu<cube>(256,256,3);
    }

    clock_t tStart = clock();

    layer::Input input(Shape(2, 16,16,3));
    layer::Conv2d c1(&input, 16, 3, Padding::SAME, 1, Activation::relu);
    layer::Conv2d c2(&c1, 32, 3, Padding::SAME, 1, Activation::sigmoid);
    layer::Conv2d c3(&c2, 32, 3, Padding::VALID, 1, Activation::relu);
    layer::Conv2d c4(&c3, 32, 3, Padding::VALID, 1, Activation::sigmoid);

    Model model(&c4);

    model.summary();
    
    // model.predict(field);
    printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
    // std::ifstream i("/home/mxw/dev/notebook/weights.dat");

    // json j_from_bson = json::from_bson(i);
    // cout << j_from_bson["root"];

    // json j;
    // i >> j;
    
    // cout << j;
    return 0;
}
