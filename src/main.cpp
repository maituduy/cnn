#include <iostream>
#include <armadillo>
#include <opencv2/opencv.hpp>
#include <iostream>

#include "nn_ops.h"
#include "activation.h"
#include "f.h"

using namespace std;
using namespace arma;
using namespace ops;
using namespace f;
using namespace cv;

template<class T3>
arma::Mat <T3> cvMat2armaMat(cv::Mat & cvMatIn) 
{ 
    return arma::Mat <T3> (cvMatIn.data, cvMatIn.rows, cvMatIn.cols); 
}

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

    cv::Mat image;
    image = cv::imread("/home/xmw//Desktop/1.png", CV_LOAD_IMAGE_COLOR);
    
    cv::Mat img(2, 2, 2, CV_32FC1);
    cv::randu(img, cv::Scalar(0), cv::Scalar(255));
        
    std::cout << "img Test" << img << std::endl;
    cv::Mat imgt(img.t());
        
    // mat to armadillo
    arma::fmat armaConv(imgt.ptr<float>(), 2, 2);
    std::cout << "armaConv" << std::endl << armaConv << std::endl;
    return 0;
}
