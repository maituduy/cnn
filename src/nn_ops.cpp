#include "nn_ops.h"
#include "f.h"
#include "mtype.h"
#include "armadillo"

using namespace arma;
using namespace f;
using namespace mtype;

namespace ops {   

    arma::field<cube> NnOps::conv2d(const arma::field<cube> &input, arma::field<cube> kernel, Padding padding, int stride) {
        arma::field<cube> result(input.n_elem);
        auto output_size = f::Common::get_output_size(input(0).slice(0).n_rows, padding, kernel(0).n_rows, stride);
        
        for (size_t i = 0; i < input.n_elem; i++) {
            
            arma::cube el = input(i);
            arma::cube sub_res(output_size, output_size, kernel.n_elem);

            for (size_t j = 0; j < kernel.n_elem; j++) {
                
                arma::cube products(output_size, output_size, el.n_slices);
                for (size_t k = 0; k < el.n_slices; k++) 
                    products.slice(k) = Conv2d::conv2d(
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

    arma::field<cube> NnOps::conv2d_transpose(const arma::field<cube> &input, arma::field<cube> kernel, Padding padding, int stride) {
        arma::field<cube> result(input.n_elem);
        
        mtype::Size output_size = f::Conv2d_Transpose::get_output_size(input(0).slice(0), kernel(0).n_rows, padding, stride);
        
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
        auto output_size = f::Common::get_output_size(input(0).slice(0).n_rows, padding, pooling_size, stride);

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