#include "armadillo"
#include "f.h"

using namespace arma;

namespace f {

    // ************************************Conv2d_Transpose************************************
    // ****************************************BEGIN************************************
    arma::mat Conv2d_Transpose::merge_cols(const arma::mat &a, const arma::mat &b, int stride) {
        bool comp = a.n_cols > b.n_cols;

        int left = comp ? a.n_cols + stride - b.n_cols: stride;
        int mid = comp ? b.n_cols - stride : a.n_cols - stride;
        int right = stride;

        return arma::join_rows(
            a.head_cols(left), 
            a.tail_cols(mid) + b.head_cols(mid),
            b.tail_cols(right)
        );
    }

    arma::mat Conv2d_Transpose::merge_rows(const arma::mat &a, const arma::mat &b, int stride) {
        bool comp = a.n_rows > b.n_rows;

        int left = comp ? a.n_rows + stride - b.n_rows: stride;
        int mid = comp ? b.n_rows - stride: a.n_rows - stride;
        int right = stride;

        return arma::join_cols(
            a.head_rows(left), 
            a.tail_rows(mid) + b.head_rows(mid),
            b.tail_rows(right)
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

    arma::mat Conv2d_Transpose::apply_conv2d_transpose_padding(const arma::mat &a, Position pos, int padding) {
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
        
        Size output_size = Conv2d_Transpose::get_output_size(a, kernel.n_rows, padding, stride);
        arma::mat out = arma::mat(output_size.w, output_size.h);
        
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
        
        return pad_flag ? tmp.submat(0, 0, tmp.n_rows - 2, tmp.n_cols - 2) : tmp;
    }

    Size Conv2d_Transpose::get_output_size(const arma::mat &a, int kernel_size, Padding padding, int stride) {
        return padding == Padding::SAME ? 
            Size(stride * a.n_rows, stride * a.n_cols) : 
            Size(stride * a.n_rows + std::max(kernel_size - stride, 0), stride * a.n_cols + std::max(kernel_size - stride, 0));
    }

    // ****************************************END********************************************
    // ************************************Conv2d_Transpose************************************
    

    // ************************************Max_Pooling2d*****************************************
    // ***************************************BEGIN**********************************************

    double Pooling2D::pool_sub(const arma::mat &a, PoolingMode pool_mode) {
        return (pool_mode == PoolingMode::MAX) ? a.max(): arma::mean(arma::mean(a));
    }

    arma::mat Pooling2D::pool(arma::mat a, int pooling_size, 
                            Padding padding, PoolingMode pool_mode, int stride) {
        
        int output_size = Common::get_output_size(a, padding, pooling_size, stride);

        double needed_pad = Common::get_needed_pad(a, output_size, pooling_size, stride);
        
        int min_x, min_y, max_x, max_y;

        if (padding == Padding::SAME) {
            a = Common::apply_needed_pad(a, needed_pad);

            min_x = min_y = (int)needed_pad;
            max_x = max_y = a.n_rows - 1 - (int)needed_pad;
            if ((int)needed_pad != needed_pad) {            
                max_x--;
                max_y--;
            }
        } else {
            min_x = min_y = 0;
            max_x = max_y = a.n_rows;
        }

        int index = 0;
        
        arma::vec vec_res(output_size*output_size);
        for (size_t i = 0; i < a.n_rows; i += stride)
            for (size_t j = 0; j < a.n_cols; j += stride)
                if (i + pooling_size - 1  < a.n_rows && j + pooling_size - 1 < a.n_cols) {
                    int start_x = i;
                    int start_y = j;
                    int end_x = i + pooling_size - 1;
                    int end_y = j + pooling_size - 1;

                    if (pool_mode == PoolingMode::AVERAGE_TF || pool_mode == PoolingMode::MAX) {
                        if (start_x < min_x) start_x = min_x;
                        if (start_y < min_y) start_y = min_y;
                        if (end_x > max_x) end_x = max_x;
                        if (end_y > max_y) end_y = max_y;
                    }

                    vec_res(index++) = Pooling2D::pool_sub(
                        a.submat(
                            start_x,
                            start_y, 
                            end_x, 
                            end_y
                        ), 
                        pool_mode
                    );
                }
                    
                    
        
        return arma::reshape(vec_res, output_size, output_size).t();

    }
    // *****************************************END**********************************************
    // ************************************Max_Pooling2d*****************************************

    // *******************************************Conv2d*****************************************
    // *******************************************BEGIN******************************************

    arma::mat Conv2d::conv2d(arma::mat a, const arma::mat &kernel, Padding padding, int stride){
        int output_size = Common::get_output_size(a, padding, kernel.n_rows, stride);

        double needed_pad = Common::get_needed_pad(a, output_size, kernel.n_rows, stride);

        if (padding == Padding::SAME) 
            a = Common::apply_needed_pad(a, needed_pad);

        int index = 0;
        arma::vec vec_res(output_size*output_size);
        for (size_t i = 0; i < a.n_rows; i += stride)
            for (size_t j = 0; j < a.n_cols; j += stride)
                if (i + kernel.n_rows - 1  < a.n_rows && j + kernel.n_rows - 1 < a.n_cols) 
                    vec_res(index++) = Conv2d::dot_sum(
                        a.submat(
                            i,
                            j, 
                            i + kernel.n_rows - 1, 
                            j + kernel.n_rows - 1
                        ),
                        kernel
                    );
        
        return arma::reshape(vec_res, output_size, output_size).t();
    }


    double Conv2d::dot_sum(arma::mat a, const arma::mat &kernel) {

        for (size_t i = 0; i < a.n_rows; i++) 
            for (size_t j = 0; j < a.n_cols; j++)
                a(i,j) *= kernel(i,j);

        return arma::accu(a);
        
    }
    // *******************************************Conv2d*****************************************
    // *******************************************END********************************************

    // *******************************************Common*****************************************
    // *******************************************BEGIN******************************************

    arma::mat Common::pad(const arma::mat &a, PaddingShape paddings) {
        arma::mat result(a);
        // top
        result.insert_rows(0, paddings.t);
        // bot
        result.insert_rows(result.n_rows, paddings.b);
        // left
        result.insert_cols(0, paddings.l);
        // right
        result.insert_cols(result.n_cols, paddings.r);
        return result;
    }

    arma::mat Common::pad(const arma::mat &a, int padding) {
        return Common::pad(a, PaddingShape(padding, padding, padding, padding));
    }

    int Common::get_output_size(const arma::mat &a, Padding padding, int kernel_size, int stride) {
        return padding == Padding::SAME ? 
                ceil((float)a.n_rows / stride) : 
                (a.n_rows - kernel_size) / stride + 1;
    }

    double Common::get_needed_pad(const arma::mat &a, int output_size, int kernel_size, int stride) {
        return std::max(((double)stride * (output_size - 1) - a.n_rows + kernel_size) / 2, 0.0);
    }

    arma::mat Common::apply_needed_pad(arma::mat a, double needed_pad) {
           
        a = Common::pad(a, (int)needed_pad);
        if ((int)needed_pad != needed_pad) 
            a = Common::pad(a, PaddingShape(0,1,0,1));
        
        return a;
    }
    // *********************************************END******************************************
    // *******************************************Common*****************************************
}