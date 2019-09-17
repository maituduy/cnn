#include "model.h"
    
Model::Model(){};

Model::Model(Layer *output_layer) {
    this->output_layer = output_layer;
    Model::separate();
};

void Model::separate() {
    Layer* last = output_layer;
    int i = 0;

    while (last) {
        this->layers.insert(this->layers.begin(), last);
        last = last->get_pre_layer();
    }

    // create names
    for (auto it = ++this->layers.begin(); it != this->layers.end(); ++it) {
        const char* class_name = (*it)->classname();

        if (this->counter.find(class_name) == this->counter.end())
            this->counter[class_name] = 1;
        else 
            this->counter[class_name]++;
        
        (*it)->set_attr("name", class_name + std::string("-") + std::to_string(this->counter[class_name]));            
    }
        
}

void Model::summary() {
    for (auto it = ++this->layers.begin(); it != this->layers.end(); ++it)
        (*it)->display_config();
}


arma::field<arma::cube> Model::predict(arma::field<arma::cube> input) {
    (*this->layers.begin())->set_output(input);

    for (auto it = ++this->layers.begin(); it != this->layers.end(); ++it)
        (*it)->foward();
}
