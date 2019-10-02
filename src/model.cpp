#include "model.h"
#include "layers/input.h"

Model::Model(){};

Model::Model(Layer *output_layer) {
    this->output_layer = output_layer;
}

void Model::separate() {
    Layer* last = this->output_layer;
    while (last) {
        this->layers.insert(this->layers.begin(), last);
        last = last->get_pre_layer();
    }

    for (auto it = ++this->layers.begin(); it != this->layers.end(); ++it) {
        const char *class_name = (*it)->classname();
        if (this->counter.find(class_name) == this->counter.end())
            this->counter[class_name] = 1;
        else
            this->counter[class_name]++;

        (*it)->set_attr("name", class_name + std::string("-") + std::to_string(this->counter[class_name]));

        if ((*it)->get_pre_layer() != nullptr) {
            (*it)->set_input((*it)->get_pre_layer()->get_output());
            (*it)->set_attr("input_shape", (*it)->get_pre_layer()->get_attr<Shape>("output_shape"));
        }
        (*it)->initialize_config();
        (*it)->initialize_weights();
    }
}

void Model::summary() {
    for (auto it = ++this->layers.begin(); it != this->layers.end(); ++it)
        (*it)->display_config();
}


arma::field<arma::cube> &Model::predict(arma::field<arma::cube> input) {
    this->layers[0]->set_output(input);
     for (auto it = ++this->layers.begin(); it != this->layers.end(); ++it)
         (*it)->foward();
        
    return (this->layers.back())->get_output();
}

void Model::load_weights(const std::string& path) {

    std::ifstream input(path, std::ios_base::binary);
    json j_from_bson = json::from_bson(input);

    int j = 1;
    int count = 0;
    wtype tmp;
    for (json::iterator it = j_from_bson["root"].begin(); it != j_from_bson["root"].end(); ++it) {
//        std::cout << (*it) << "\n";
        if ((*it)[0].is_array()) {

            Shape shape = this->layers[j]->get_attr<Shape>("kernel_shape");
            arma::field<arma::cube> kernel(shape.batch);
            Parser::parse_arma(it, &kernel, shape);
            
            tmp.push_back(kernel);
        }
        else {
            std::vector<double> v;
            for (auto & i : (*it))
                v.push_back(i);
            
            tmp.push_back(arma::vec(v));
        }

        count++;
        if (count == this->layers[j]->get_weights().size()) {
            this->layers[j]->set_weights(tmp);
            j++;
            if (this->layers.size() <= j || it+1 == j_from_bson["root"].end()) break;
            while (!this->layers[j]->check_weights()) j++;
            count = 0;
            tmp.clear();
        }
    }

}

std::vector<Layer*> &Model::get_layers() {
    return this->layers;
}

Model *Model::add(const Layer& tmp_layer) {
    auto tmp = tmp_layer.clone();

    if (this->output_layer != nullptr) {
        *tmp << this->output_layer;
        this->output_layer = tmp;
    }
    else
        this->output_layer = tmp;
    return this;
}


Model *Model::add(Layer* tmp_layer) {
    if (this->output_layer != nullptr) {
        *tmp_layer << this->output_layer;
        this->output_layer = tmp_layer;
    }
    else
        this->output_layer = tmp_layer;
    return this;
}

Model *Model::sign(const std::string& id) {
    this->ids[id] = this->output_layer;
    return this;
}

Layer *Model::get(const std::string& id) {
    if (this->ids.find(id) == this->ids.end())
        throw "ID not found";
    return this->ids[id];
}
