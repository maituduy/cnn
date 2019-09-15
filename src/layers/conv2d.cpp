#include "armadillo"
#include "layer.h"
#include "activation.h"
#include "f.h"
#include "nn_ops.h"

using namespace ops;

namespace layer {

    class Conv2d: public Layer {
        Layer pre_layer;
        Dict config;
        arma::field<arma::cube> kernel;
        arma::field<arma::cube> input;
        arma::field<arma::cube> output;
        double (*activation)(double);

        public:
            Conv2d( const Layer &layer,
                    int n_filters, 
                    int kernel_size, 
                    Padding padding = Padding::SAME,
                    int stride = 1, 
                    double (*activation)(double) = nullptr) {
                
                this->pre_layer = layer;
                
                Shape input_shape = std::get<Shape>(this->pre_layer.get_config()["output_shape"]);
                
                int output_size = f::Common::get_output_size(input_shape.w, padding, kernel_size, stride);
                
                Shape output_shape = 
                    Shape(
                        input_shape.batch,
                        output_size,
                        output_size,
                        n_filters
                    );

                this->config["input_shape"] = input_shape;
                this->config["output_shape"] = output_shape;

                this->config["kernel_shape"] = 
                    Shape(n_filters, kernel_size, kernel_size, input_shape.c);
                
                this->config["stride"] = stride;
                this->config["padding"] = padding;
                this->activation = activation;

                Layer::set_config(this->config);

           }

            void initialize() {
                Shape kernel_shape = this->get_attr<Shape>("kernel_shape");
                kernel = arma::field<cube>(kernel_shape.batch);

                for (size_t i = 0; i < kernel_shape.batch; i++) 
                    kernel(i) = arma::randu<cube>(kernel_shape.w, kernel_shape.h, kernel_shape.c);
                
                this->kernel = kernel;
            }

            void foward() {
                this->input = this->pre_layer.get_output();
                clock_t tStart = clock();

                this->output = 
                    ops::NnOps::conv2d(
                        this->input, 
                        this->kernel, 
                        get_attr<Padding>("padding"),
                        get_attr<int>("stride")
                    );
                
                if (this->activation)
                    Activation::active(this->output, this->activation);
                
                printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
            
                Layer::set_output(this->output);

            }
    };
}