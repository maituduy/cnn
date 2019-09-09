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
            static arma::mat conv2d_transpose(const arma::mat &a, const arma::mat &kernel, Padding padding = Padding::SAME, int stride=1);
    };

    
}