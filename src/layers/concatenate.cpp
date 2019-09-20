#include "layer.h"
#include <initializer_list>

namespace layer {
    class Concatenate: public Layer {
        std::vector<arma::field<arma::cube>*> list;
        public:
            Concatenate(
                Layer* pre_layer,
                std::initializer_list<Layer*> list
            
            ): Layer(pre_layer, false) {
                
                int out_channel = 0;
                for (auto elem: list) {
                    this->list.push_back(&(*elem).get_output());
                    out_channel += elem->get_attr<Shape>("output_shape").c;
                }
                auto it = list.begin();
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