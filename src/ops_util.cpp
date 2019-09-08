#include <armadillo>
#include "../include/ops_util.h"

using namespace arma;

namespace ops_util {
    arma::mat merge_cols(const arma::mat &a, const arma::mat &b) {
        return arma::join_cols(
            a.head_cols(1), 
            a.tail_cols(a.n_cols - 1) + b.head_rows(b.n_cols - 1),
            b.tail_cols(1)
        );
    }
}