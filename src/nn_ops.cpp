#include "nn_ops.h"
#include "armadillo"

using namespace arma;

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

    static arma::field<cube> conv2d_transpose(const arma::field<cube> &input, arma::field<cube> kernel, Padding padding, int stride) {
        arma::field<cube> result(input.n_elem);

        for (size_t i = 0; i < input.n_elem; i++) {
            
            arma::cube el = input(i);
            
            for (size_t j = 0; j < kernel.n_elem; j++) {
                arma::cube tmp;
                for (size_t k = 0; k < el.n_slices; k++) {
                
                    el.slice(j);
                }
            }
            
            
        }

        return result;
    }
}