#include <armadillo>
#include "mtype.h"

using namespace arma;
using namespace mtype;

namespace ops_util {
    
    class Conv2d_Transpose {
        public:
            static arma::mat merge_cols(const arma::mat &a, const arma::mat &b, int stride=1);
            static arma::mat merge_rows(const arma::mat &a, const arma::mat &b, int stride=1);
            static arma::mat apply_conv2d_transpose_padding(const arma::mat &a, int padding);
            static arma::mat apply_conv2d_transpose_padding_by_pos(const arma::mat &a, Position pos, int padding);
            static mtype::Size get_output_size(const arma::mat &a, int kernel_size, Padding padding, int stride=1);
            static arma::mat conv2d_transpose(const arma::mat &a, const arma::mat &kernel, Padding padding = Padding::SAME, int stride=1);
    };

    class Common {
        public:
            static arma::mat pad(const arma::mat &a, int padding_size = 1);
            static arma::mat pad_by_pos(const arma::mat &a, Position pos, int padding_size = 1);
    };
    
    class Max_Pooling2D {
        public:
            static arma::mat max_pooling(const arma::mat &a, int pooling_size=2, Padding padding = Padding::SAME, int stride=1);
    };

    
}