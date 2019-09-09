#include "armadillo"
#include "ops_util.h"

using namespace arma;

namespace ops_util {

    // ************************************Conv2d_Transpose************************************

    arma::mat Conv2d_Transpose::merge_cols(const arma::mat &a, const arma::mat &b, int stride) {

        return arma::join_rows(
            a.head_cols(stride), 
            a.tail_cols(a.n_cols - stride) + b.head_cols(b.n_cols - stride),
            b.tail_cols(stride)
        );
    }

    arma::mat Conv2d_Transpose::merge_rows(const arma::mat &a, const arma::mat &b, int stride) {

        return arma::join_cols(
            a.head_rows(stride), 
            a.tail_rows(a.n_rows - stride) + b.head_rows(b.n_rows - stride),
            b.tail_rows(stride)
        );
    }

    arma::mat Conv2d_Transpose::apply_conv2d_transpose_padding(const arma::mat &a, int padding) {

        return a.submat(
            padding,
            padding,
            a.n_rows - padding - 1,
            a.n_cols - padding - 1
        );
    }

    arma::mat Conv2d_Transpose::apply_conv2d_transpose_padding_by_pos(const arma::mat &a, Position pos, int padding) {
        switch (pos) {

        case Position::LEFT:
            return a.tail_cols(a.n_cols - padding);
        
        case Position::RIGHT:
            return a.head_cols(a.n_cols - padding);
        
        case Position::TOP:
            return a.tail_rows(a.n_rows - padding);

        default:
            return a.head_rows(a.n_rows - padding);
        }
    }

    arma::mat Conv2d_Transpose::conv2d_transpose(const arma::mat &a, const arma::mat &kernel, Padding padding, int stride) {
        
        float padding_size = padding == Padding::SAME ? (float)(kernel.n_rows - stride)/2 : 0; 
        
        bool pad_flag = ((int)(padding_size * 2) % 2 == 0) ? false : true;
        
        arma::mat out = arma::mat(
            stride * (a.n_rows -1) + kernel.n_rows - 2 * padding_size,
            stride * (a.n_cols -1) + kernel.n_cols - 2 * padding_size
        );

        arma::field<arma::mat> tmp_rows(a.n_rows);
        for (size_t i = 0; i < a.n_rows; i++) {

            arma::mat tmp_cols = a(i,0) * kernel;

            for (size_t j = 1; j < a.n_cols; j++) 
                tmp_cols = Conv2d_Transpose::merge_cols(tmp_cols, a(i,j) * kernel, stride);                
            
            tmp_rows(i) = tmp_cols;  
              
        }
        
        arma::mat tmp = tmp_rows(0);
        for (size_t i = 1; i < tmp_rows.n_elem; i++)
            tmp = Conv2d_Transpose::merge_rows(tmp, tmp_rows(i), stride);
        
        tmp = Conv2d_Transpose::apply_conv2d_transpose_padding(tmp, (int) padding_size);
        
        return pad_flag ? tmp.submat(1, 1, tmp.n_rows - 1, tmp.n_cols - 1) : tmp;
    }

    // ************************************Conv2d_Transpose************************************
}