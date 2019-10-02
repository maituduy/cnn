#include "parser.h"

namespace parser {

    void Parser::parse_arma(json::iterator &json, arma::field<arma::cube> *kernel, Shape shape) {
        int batch = 0;
        for (json::iterator i1 = (*json).begin(); i1 != (*json).end(); ++i1) {
            arma::cube cube(shape.w, shape.h, shape.c);

            int r =0;
            for (json::iterator i2 = (*i1).begin(); i2 != (*i1).end(); ++i2, ++r) {
                int c = 0;
                for (json::iterator i3 = (*i2).begin(); i3 != (*i2).end(); ++i3, ++c) {
                    int slice = 0;
                    for (json::iterator i4 = (*i3).begin(); i4 != (*i3).end(); ++i4, ++slice) 
                        cube(r,c,slice) = *i4;
                        
                } 
            }
            (*kernel)(batch++) = cube;
        }
    }

    arma::field<arma::cube> Parser::get_input(const std::string& path) {
        std::ifstream input(path, std::ios_base::binary);
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

}
