#include "layer.h"

namespace layer {

    class Conv2d: public Layer {
        double (*activation)(double);

        public:
            Conv2d( 
                Layer *layer,
                int n_filters, 
                int kernel_size, 
                Padding padding = Padding::SAME,
                int stride = 1, 
                double (*activation)(double) = nullptr
                
            ): Layer(layer, true) {
                Shape input_shape = this->get_attr<Shape>("input_shape");
                
                int output_size = f::Common::get_output_size(input_shape.w, padding, kernel_size, stride);
                
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
                this->activation = activation;

                this->initialize();
           }

            void foward() {
                this->output = 
                    ops::NnOps::conv2d(
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
                    ops::Activation::active(this->output, this->activation);
            
                this->set_output(this->output);

            }

            const char* classname() { return "Conv2d";}
    };
}