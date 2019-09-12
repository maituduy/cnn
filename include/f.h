#include "armadillo"
#include "mtype.h"

using namespace arma;
using namespace mtype;

namespace f {
    
    class Conv2d_Transpose {
        public:
            static arma::mat merge_cols(const arma::mat &a, const arma::mat &b, int stride=1);
            static arma::mat merge_rows(const arma::mat &a, const arma::mat &b, int stride=1);
            static arma::mat apply_conv2d_transpose_padding(const arma::mat &a, int padding);
            static arma::mat apply_conv2d_transpose_padding(const arma::mat &a, Position pos, int padding);
            static mtype::Size get_output_size(const arma::mat &a, int kernel_size, Padding padding, int stride=1);
            static arma::mat conv2d_transpose(const arma::mat &a, const arma::mat &kernel, Padding padding = Padding::SAME, int stride=1);
    };

    class Common {
        public:
            // top bot left right
            static arma::mat pad(const arma::mat &a, PaddingShape shape);
            static arma::mat pad(const arma::mat &a, int padding);
            static int get_output_size(const arma::mat &a, Padding padding, int kernel_size, int stride);
            static double get_needed_pad(const arma::mat &a, int output_size, int kernel_size, int stride);
            static arma::mat apply_needed_pad(arma::mat a, double needed_pad);
    };
    
    class Pooling2D {
        public:
            // https://github.com/tensorflow/tensorflow/blob/3c3c0481ec087aca4fa875d6d936f19b31191fc1/tensorflow/core/framework/common_shape_fns.cc#L40-L48
            // https://github.com/pytorch/pytorch/issues/3867

            static arma::mat pool(arma::mat a, int pooling_size=2, Padding padding = Padding::SAME, PoolingMode pool_mode = PoolingMode::MAX, int stride=1);
            static double pool_sub(const arma::mat &a, PoolingMode pool_mode = PoolingMode::MAX);
    };

    class Conv2d {
        public:
            static arma::mat conv2d(arma::mat a, const arma::mat &kernel, Padding padding = Padding::SAME, int stride=1);
            static double dot_sum(arma::mat a, const arma::mat &kernel);
    };
}