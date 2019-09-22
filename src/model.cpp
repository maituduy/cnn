#include "model.h"

Model::Model(){};

Model::Model(Layer *output_layer) {
    this->output_layer = output_layer;
};

void Model::separate() {
    Layer* last = this->output_layer;
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


arma::field<arma::cube> &Model::predict(arma::field<arma::cube> input) {
    this->layers[0]->set_output(input);
    for (int i=1; i< this->layers.size(); i++) {
        this->layers[i]->foward();
    // for (auto it = ++this->layers.begin(); it != this->layers.end(); ++it)  {
    //     std::cout << 1;
    //     (*it)->foward();
    }
    std::cout<<"chay tiep o";
        
    return (this->layers.back())->get_output();
}

void Model::load_weights(std::string path) {

    std::ifstream input(path);
    json j_from_bson = json::from_bson(input);
    
    int j = 1;
    int count = 0;
    wtype tmp;
    
    for (json::iterator it = j_from_bson["root"].begin(); it != j_from_bson["root"].end(); ++it) {
         
        if ((*it)[0].is_array()) {
            Shape shape = this->layers[j]->get_attr<Shape>("kernel_shape");
            arma::field<arma::cube> kernel(shape.batch);
            Parser::parse_arma(it, &kernel, shape);
            
            tmp.push_back(kernel);
        }
        else {
            std::vector<double> v;
            for (json::iterator i = (*it).begin(); i != (*it).end(); ++i)
                v.push_back(*i);
            
            tmp.push_back(arma::vec(v));
        }
        count++;
        if (count == this->layers[j]->get_weights().size()) {
            this->layers[j]->set_weights(tmp);
            j++;
            if (this->layers.size() <= j) break;
            while (!this->layers[j]->check_weights()) j++;
            count = 0;
            tmp.clear();
        }
    }


}

std::vector<Layer*> &Model::get_layers() {
    return this->layers;
}

arma::field<arma::cube> Model::get_input(std::string path) {
    std::ifstream input(path);
    json j_from_bson = json::from_bson(input);

    int list[4];
    int i = 0;
    json shape = j_from_bson["shape"];
    for (auto it = shape.begin(); it!=shape.end(); it++, i++)
        list[i] = *it;

    Shape input_shape(list[0], list[1], list[2], list[3]);

    arma::field<arma::cube> in(input_shape.batch);

    auto it = j_from_bson["input"].begin();
    parser::Parser::parse_arma(it, &in, input_shape);
        
    return in;
}

Model *Model::add(const Layer &tmp_layer) {
    auto tmp = tmp_layer.clone();
    if (this->output_layer != nullptr) 
        this->output_layer = tmp->operator()(this->output_layer);
        
    else 
        this->output_layer = tmp;

    return this;
}

Model *Model::sign(std::string id) {
    this->ids[id] = this->output_layer;
    return this;
}

Layer *Model::get(std::string id) {
    if (this->ids.find(id) == this->ids.end())
        throw "ID not found";

    return this->ids[id];
}
