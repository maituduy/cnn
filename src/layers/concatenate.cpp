#include "concatenate.h"

namespace layer {
    Concatenate::Concatenate(std::initializer_list<Layer*> list): Layer(false){init_list = list;};

    Concatenate::Concatenate(const Concatenate& layer): Layer(layer) {
        this->list = layer.list;
        this->init_list = layer.init_list;
    }
    Layer* Concatenate::clone() const{
        return new Concatenate(*this);
    }

    void Concatenate::initialize_config() {
                                
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

    void Concatenate::foward() {
        this->output = f::Common::concatenate(list);
    }

    const char* Concatenate::classname() { 
        return "Concatenate";
    }
}