#pragma once

#include "armadillo"
#include "mtype.h"
#include <string>
#include <time.h>

using namespace mtype;

namespace layer {

    class Layer {
        protected:

            Layer *pre_layer;
            arma::field<arma::cube> input, kernel, output;
            arma::Row<double> bias;
            Dict config;
            bool has_weights= true;

        
        public:
            Layer(){};

            Layer(Layer *pre_layer, bool has_weights) {
                this->pre_layer = pre_layer;
                this->has_weights = has_weights;
            };
            
            virtual ~Layer(){};
            virtual void foward(){};
            virtual const char* classname() { return "Layer";}

            void initialize() {
                Shape kernel_shape = this->get_attr<Shape>("kernel_shape");
                Shape output_shape = this->get_attr<Shape>("output_shape");

                kernel = arma::field<arma::cube>(kernel_shape.batch);

                for (size_t i = 0; i < kernel_shape.batch; i++) 
                    kernel(i) = arma::randu<arma::cube>(kernel_shape.w, kernel_shape.h, kernel_shape.c);
                
                this->kernel = kernel;

                this->bias = bias.randu(output_shape.c);
            }
            
            void set_kernel(arma::field<arma::cube> &kernel){
                this->kernel = kernel;
            };

            void set_bias(arma::Row<double> &bias) {
                this->bias = bias;
            };

            arma::field<arma::cube> &get_kernel() {
                return this->kernel;
            }

            arma::Row<double> &get_bias() {
                return this->bias;
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

            void set_config(const Dict &config) {
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

    };
}