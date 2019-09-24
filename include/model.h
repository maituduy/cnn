#pragma once

#include "layer.h"
#include "parser.h"
#include <vector>
#include "armadillo"
#include <map>
#include <any>
#include <json.hpp>

using json = nlohmann::json;
using namespace layer;
using namespace parser;

class Model{
    private:
        Layer *output_layer = nullptr;
        std::vector<Layer*> layers;
        std::map<std::string, int> counter;
        std::map<std::string, Layer*> ids;

    public:
        Model();
        Model(Layer *output_layer);
        void separate();
        void summary();
        arma::field<arma::cube> &predict(arma::field<arma::cube> input);
        void load_weights(const std::string& path);
        std::vector<Layer*> &get_layers();
        Model *add(const Layer&);
        Model *add(Layer*);
        Model *sign(const std::string&);
        Layer *get(const std::string&);
};
