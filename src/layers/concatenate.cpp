#include "layer.h"
#include <initializer_list>

namespace layer {
    class Concatenate: public Layer {
        std::vector<arma::field<arma::cube>*> list;
        std::initializer_list<Layer*> init_list;
        public:
            Concatenate(std::initializer_list<Layer*> list): Layer(false){init_list = list;};

            void initialize_config() {
                                
                int out_channel = 0;
                for (auto elem: init_list) {
                    this->list.push_back(&(*elem).get_output());
                    out_channel += elem->get_attr<Shape>("output_shape").c;
                }
                auto it = init_list.begin();
                Shape tmp = (*it)->get_attr<Shape>("output_shape");
                tmp.c = out_channel;
                config["output_shape"] = tmp;
            }

            void foward() {
                this->output = f::Common::concatenate(list);
            }

            const char* classname() { return "Concatenate";}
    };
}