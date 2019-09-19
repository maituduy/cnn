#pragma once

#include "layer.h"
#include "parser.h"
#include <vector>
#include "armadillo"
#include <map>
#include <json.hpp>

using json = nlohmann::json;
using namespace layer;
using namespace parser;

class Model{
    private:
        Layer *output_layer;
        std::vector<Layer*> layers;
        std::map<std::string, int> counter;    

    public:
        Model();
        Model(Layer *output_layer);
        void separate();
        void summary();
        arma::field<arma::cube> &predict(arma::field<arma::cube> input);
        void load_weights(std::string path);
        std::vector<Layer*> &get_layers();
};
