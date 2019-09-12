#include "nn_ops.h"
#include "f.h"
#include "mtype.h"
#include "armadillo"

using namespace arma;
using namespace f;
using namespace mtype;

namespace ops {   

    // arma::field<cube> NnOps::conv2d(const arma::field<arma::cube> &input, arma::field<arma::cube> kernel, std::string shape) {
        // in : 8, 256, 256 , 32
        // kernel : 64, 3, 3, 32
        // out: 8, 256, 256, 64
        
        // arma::field<arma::cube> result(input.n_elem);
        
        // for (size_t i = 0; i < input.n_elem; i++) {
        //     cube el = input[i];
        //     cube sub_res = 0
        //     for (size_t j = 0; j < kernel.n_elem; j++){
        //         cube sub_kenel = kernel(j);
                

        //         cube tmp;
        //         tmp.copy_size(el);
                
        //         for (size_t k = 0; k < sub_kenel.n_slices; k++) {
        //             tmp.slice(k) = arma::conv2(el.slice(k) ,sub_kenel.slice(k));
        //         }
                
        //         result(i) = 
        //     }
            
            // cube el = input[i];
            // int n_slices = el.n_elem_slice;
            // cube tmp = arma::zeros(el.n_rows, el.n_cols, el.n_slices);
            
            // for (size_t j = 0; j < n_slices; i++){
            //     tmp.slice(j) = arma::conv2();
            // }
             
        // }
    // }

    arma::field<cube> NnOps::conv2d_transpose(const arma::field<cube> &input, arma::field<cube> kernel, Padding padding, int stride) {
        arma::field<cube> result(input.n_elem);
        
        mtype::Size output_size = f::Conv2d_Transpose::get_output_size(input(0).slice(0), kernel.n_rows, padding, stride);
        
        for (size_t i = 0; i < input.n_elem; i++) {
            
            arma::cube el = input(i);
            arma::cube sub_res(output_size.w, output_size.h, kernel.n_elem);

            for (size_t j = 0; j < kernel.n_elem; j++) {
                
                arma::cube products(output_size.w, output_size.h, el.n_slices);
                for (size_t k = 0; k < el.n_slices; k++) 
                    products.slice(k) = Conv2d_Transpose::conv2d_transpose(
                        el.slice(k), 
                        kernel(j).slice(k),
                        padding,
                        stride
                    );

                sub_res.slice(j) = arma::sum(products, 2);
            }
            
            result(i) = sub_res;
        }

        return result;
    }
    
    arma::field<cube> NnOps::pooling2d(const arma::field<cube> &input, int pooling_size, Padding padding, PoolingMode mode, int stride) {
        arma::field<cube> result(input.n_elem);
        auto output_size = f::Common::get_output_size(input(0).slice(0), padding, pooling_size, stride);

        for (size_t i = 0; i < input.n_elem; i++) {
            auto el = input(i);
            arma::cube sub_res(output_size, output_size, el.n_slices);

            for (size_t j = 0; j < el.n_slices; j++) 
                sub_res.slice(j) = f::Pooling2D::pool(el.slice(j), pooling_size, padding, mode, stride) ;    
            
            result(i) = sub_res;
        }
        
        return result;
    }
}