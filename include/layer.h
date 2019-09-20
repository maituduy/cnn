#pragma once

#include "armadillo"
#include "mtype.h"
#include <string>
#include <time.h>
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

            Layer(Layer *pre_layer, bool has_weights) {
                this->pre_layer = pre_layer;
                this->has_weights = has_weights;
                if (pre_layer != nullptr) {
                    this->input = &pre_layer->output;
                    config["input_shape"] = pre_layer->get_attr<Shape>("output_shape");
                }
            };
            
            virtual ~Layer(){};
            virtual void foward(){};
            virtual const char* classname() { return "Layer";}

            virtual void initialize() {
                Shape kernel_shape = this->get_attr<Shape>("kernel_shape");
                Shape output_shape = this->get_attr<Shape>("output_shape");

                auto kernel = arma::field<arma::cube>(kernel_shape.batch);

                for (size_t i = 0; i < kernel_shape.batch; i++) 
                    kernel(i) = arma::randu<arma::cube>(kernel_shape.w, kernel_shape.h, kernel_shape.c);
                
                this->weights.push_back(kernel);
                this->weights.push_back(arma::zeros<arma::vec>(output_shape.c));

            }
            
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

    };
}