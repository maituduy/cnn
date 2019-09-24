#include "conv2d_transpose.h"

namespace layer {

    Conv2dTranspose::Conv2dTranspose(
        int n_filters, 
        int kernel_size, 
        Padding padding,
        int stride, 
        Func activation

    ): Layer(true) {
        
        this->n_filters = n_filters;
        this->kernel_size = kernel_size;
        this->padding = padding;
        this->stride = stride;
        this->activation = activation;
    }
    
    Conv2dTranspose::Conv2dTranspose(const Conv2dTranspose& layer): Layer(layer) {
        this->n_filters = layer.n_filters;
        this->kernel_size = layer.kernel_size;
        this->padding = layer.padding;
        this->stride = layer.stride;
        this->activation = layer.activation;
        
    }

    Layer* Conv2dTranspose::clone() const {
        return new Conv2dTranspose(*this);
    }

    void Conv2dTranspose::initialize_weights() {
        
        Shape kernel_shape = this->get_attr<Shape>("kernel_shape");
        Shape output_shape = this->get_attr<Shape>("output_shape");

        auto kernel = arma::field<arma::cube>(kernel_shape.batch);

        for (size_t i = 0; i < kernel_shape.batch; i++) 
            kernel(i) = arma::randu<arma::cube>(kernel_shape.w, kernel_shape.h, kernel_shape.c);
        
        this->weights.push_back(kernel);
        this->weights.push_back(arma::zeros<arma::vec>(output_shape.c));
    }

    void Conv2dTranspose::initialize_config() {
        
        Shape input_shape = this->get_attr<Shape>("input_shape");
        int output_size = f::Conv2d_Transpose::get_output_size(input_shape.w, kernel_size, padding, stride).w;
        
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
        config["padding"] = padding;
    }

    void Conv2dTranspose::foward() {
        
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
        f::Activation::active(&this->output, this->activation);
        
        this->set_output(this->output);

    }

    const char* Conv2dTranspose::classname() { 
        return "Conv2dTranspose";
    }
}