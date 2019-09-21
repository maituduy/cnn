#pragma once

#include "armadillo"
#include "mtype.h"
#include <string>
#include "activation.h"
#include "f.h"
#include "nn_ops.h"

using namespace mtype;

typedef std::vector<std::variant<arma::field<arma::cube>, arma::vec>> wtype;

namespace layer {

    class Layer {
        protected:

            Layer *pre_layer;
            arma::field<arma::cube>* input;
            arma::field<arma::cube> output;
            wtype weights;
            Dict config;
            bool has_weights= true;

        public:
            Layer(){};
            Layer(bool has_weights): has_weights(has_weights){};

            Layer *operator()(Layer *pre_layer) {
                
                this->pre_layer = pre_layer;
                if (pre_layer != nullptr) {                
                    this->input = &this->pre_layer->output;
                    config["input_shape"] = pre_layer->get_attr<Shape>("output_shape");
                }
                initialize_config();
                initialize_weights();
                return this->clone();
            }
            
            arma::field<arma::cube> &get_input() {
                return *this->input;
            }
            
            virtual ~Layer(){};
            virtual void foward(){};
            virtual const char* classname() { return "Layer";}

            virtual void initialize_config(){};
            virtual void initialize_weights(){};

            void set_weights(wtype &weights) {
                this->weights = weights;
                
            }

            wtype &get_weights() {
                return this->weights;
            }
            
            void display_config() {
                std::cout << "{\n";
                
                for(auto& el: config){
                    std::string key = el.first;
                    std::visit(
                        [&key](auto& value){
                            std::cout << "\t" << key << ": " << value << "\n";
                        },
                        el.second // k_v.second to print all values in d
                    );
                }

                std::cout << "}\n";
            }

            Dict &get_config() {
                return config;
            }

            void set_config(Dict &config) {
                this->config = config;
            }

            arma::field<arma::cube> &get_output() {
                return this->output;
            }

            void set_output(arma::field<arma::cube> &output) {
                this->output = output;
            }

            bool check_weights() {
                return this->has_weights;
            }

            Layer *get_pre_layer() {
                return this->pre_layer;
            }
            
            template<typename T>
            T &get_attr(std::string name) {
                return std::get<T>(this->config[name]);
            }
            
            template<typename T>
            void set_attr(std::string name, T value) {
                this->config[name] = value;
            }

            Layer(const Layer &layer){
                this->pre_layer = layer.pre_layer;
                this->input = layer.input;
                this->output = layer.output;
                this->weights = layer.weights;
                this->config = layer.config;
                this->has_weights = layer.has_weights;
            };

            virtual Layer* clone() const {
                return new Layer(*this);
            }
    };
}