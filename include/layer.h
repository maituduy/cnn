#pragma once

#include "armadillo"
#include "mtype.h"
#include <string>
#include <time.h>

using namespace mtype;

namespace layer {

    class Layer {
        protected:
            arma::field<arma::cube> input, kernel, output;
            arma::vec bias;
            Dict config;

        
        public:
            Layer(){};
            
            virtual ~Layer(){};
            virtual void foward(){};
            virtual void initialize(){};

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

            Dict get_config() {
                return config;
            }

            void set_config(const Dict &config) {
                this->config = config;
            }

            arma::field<arma::cube> get_output() {
                return this->output;
            }

            void set_output(const arma::field<arma::cube> &output) {
                this->output = output;
            }

            template<typename T>
            T get_attr(std::string name) {
                return std::get<T>(this->config[name]);
            }
    };
}