#include "layer.h"
#include <vector>
#include "armadillo"
#include <map>

using namespace layer;

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
        arma::field<arma::cube> predict(arma::field<arma::cube> input);
};
