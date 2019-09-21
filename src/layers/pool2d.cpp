#include <layer.h>

namespace layer {

    class Pooling2d: public Layer {
        int kernel_size;
        PoolingMode *pooling_mode;
        Padding *padding;
        int stride;

        public:
            Pooling2d(
                int kernel_size,
                PoolingMode pooling_mode = PoolingMode::MAX,
                Padding padding = Padding::SAME,
                int stride = 1
            ): Layer(false) {
                this->kernel_size = kernel_size;
                this->pooling_mode = &pooling_mode;
                this->padding = &padding;
                this->stride = stride;
            }

            void initialize_config() {
                Shape input_shape = this->get_attr<Shape>("input_shape");
                
                int output_size = f::Common::get_output_size(input_shape.w, *padding, kernel_size, stride);
                
                Shape output_shape = 
                    Shape(
                        input_shape.batch,
                        output_size,
                        output_size,
                        input_shape.c
                    );

                config["input_shape"] = input_shape;
                config["output_shape"] = output_shape;
                config["kernel_size"] = kernel_size;
                config["stride"] = stride;
                config["padding"] = *padding;
                config["pooling_mode"] = *pooling_mode;
            }

            void foward() {
                auto result = 
                    ops::NnOps::pooling2d(
                        *this->input,
                        get_attr<int>("kernel_size"),
                        get_attr<Padding>("padding"),
                        get_attr<PoolingMode>("pooling_mode"),
                        get_attr<int>("stride")
                    );

                this->set_output(result);
            }

            const char* classname() { return "Pooling2d";}
    };
}