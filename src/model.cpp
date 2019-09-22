#include "model.h"
#include "layers/input.cpp"

Model::Model(){};

Model::Model(Layer *output_layer) {
    this->output_layer = output_layer;
    // Model::separate();
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
    std::cout << this->output_layer << "\n";
    for (auto it = ++this->layers.begin(); it != this->layers.end(); ++it)
        (*it)->display_config();
}


arma::field<arma::cube> &Model::predict(arma::field<arma::cube> input) {
    this->layers[0]->set_output(input);
    for (auto it = ++this->layers.begin(); it != this->layers.end(); ++it)  {
        (*it)->foward();
    }
        
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

            // std::get<arma::field<arma::cube>>(this->layers[j]->get_weights()[0]).print();
            // std::get<arma::field<arma::cube>>(tmp[0]).print();
            
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

    if (this->output_layer)
        this->output_layer = tmp_layer.clone()->operator()(this->output_layer);
    else 
        this->output_layer = tmp_layer.clone();
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
