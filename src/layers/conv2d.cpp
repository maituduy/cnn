#include "armadillo"
#include "layer.h"
#include "activation.h"
#include "f.h"
#include "nn_ops.h"

using namespace ops;

namespace layer {

    class Conv2d: public Layer {
        double (*activation)(double);

        public:
            Conv2d( Layer *layer,
                    int n_filters, 
                    int kernel_size, 
                    Padding padding = Padding::SAME,
                    int stride = 1, 
                    double (*activation)(double) = nullptr): Layer(layer, true){
                
                Dict config;
                
                Shape input_shape = std::get<Shape>(this->get_pre_layer()->get_config()["output_shape"]);
                
                int output_size = f::Common::get_output_size(input_shape.w, padding, kernel_size, stride);
                
                Shape output_shape = 
                    Shape(
                        input_shape.batch,
                        output_size,
                        output_size,
                        n_filters
                    );

                config["input_shape"] = input_shape;
                config["output_shape"] = output_shape;

                config["kernel_shape"] = 
                    Shape(n_filters, kernel_size, kernel_size, input_shape.c);
                
                config["stride"] = stride;
                config["padding"] = padding;
                this->activation = activation;

                this->set_config(config);
                this->initialize();
           }

            const char* classname() { return "Conv2d";}

            void foward() {
                this->input = this->get_pre_layer()->get_output();
                clock_t tStart = clock();

                this->output = 
                    ops::NnOps::conv2d(
                        this->input, 
                        this->kernel, 
                        get_attr<Padding>("padding"),
                        get_attr<int>("stride")
                    );
                
                for (size_t i = 0; i < this->output.n_elem; i++) {
                    auto el = this->output(i);
                    for (size_t j = 0; j < el.n_slices; j++) 
                        el.slice(j) += this->bias(j);
                    
                    this->output(i) = el;
                }
                
                if (this->activation)
                    Activation::active(this->output, this->activation);
                
                printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
            
                this->set_output(this->output);

            }
    };
}