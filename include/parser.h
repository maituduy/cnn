#pragma once

#include <json.hpp>
#include <armadillo>
#include "mtype.h"

using json = nlohmann::json;
using namespace mtype;

namespace parser {
    class Parser {
        public:
            static void parse_arma(json::iterator &json, arma::field<arma::cube> *kernel, Shape shape);
    };

}
