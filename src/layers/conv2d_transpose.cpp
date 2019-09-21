#include "layer.h"

namespace layer {

    class Conv2dTranspose: public Layer {

        int n_filters, kernel_size;
        Padding *padding;
        int stride;
        mtype::Activation *activation;
        
        public:
            Conv2dTranspose(
                int n_filters, 
                int kernel_size, 
                Padding padding = Padding::SAME,
                int stride = 1, 
                mtype::Activation activation = mtype::Activation::NONE
                
            ): Layer(true) {
                
                this->n_filters = n_filters;
                this->kernel_size = kernel_size;
                this->padding = &padding;
                this->stride = stride;
                this->activation = &activation;
            }
            
            void initialize_weights() {
                
                Shape kernel_shape = this->get_attr<Shape>("kernel_shape");
                Shape output_shape = this->get_attr<Shape>("output_shape");

                auto kernel = arma::field<arma::cube>(kernel_shape.batch);

                for (size_t i = 0; i < kernel_shape.batch; i++) 
                    kernel(i) = arma::randu<arma::cube>(kernel_shape.w, kernel_shape.h, kernel_shape.c);
                
                this->weights.push_back(kernel);
                this->weights.push_back(arma::zeros<arma::vec>(output_shape.c));
            }

            void initialize_config() {
                
                Shape input_shape = this->get_attr<Shape>("input_shape");
                int output_size = f::Conv2d_Transpose::get_output_size(input_shape.w, kernel_size, *padding, stride).w;
                
                Shape output_shape = 
                    Shape(
                        input_shape.batch,
                        output_size,
                        output_size,
                        n_filters
                    );

                config["output_shape"] = output_shape;

                config["kernel_shape"] = 
                    Shape(n_filters, kernel_size, kernel_size, input_shape.c);
                
                config["stride"] = stride;
                config["padding"] = *padding;
                this->activation = activation;
            }

            void foward() {
                this->output = 
                    ops::NnOps::conv2d_transpose(
                        *this->input, 
                        std::get<arma::field<arma::cube>>(this->get_weights()[0]), 
                        get_attr<Padding>("padding"),
                        get_attr<int>("stride")
                    );
                
                for (size_t i = 0; i < this->output.n_elem; i++) {
                    auto el = this->output(i);
                    for (size_t j = 0; j < el.n_slices; j++) 
                        el.slice(j) += std::get<arma::vec>(this->get_weights()[1])(j);
                    
                    this->output(i) = el;
                }
                
                if (this->activation)
                    ops::Activation::active(&this->output, *this->activation);
            
                this->set_output(this->output);

            }

            const char* classname() { return "Conv2dTranspose";}
    };
}