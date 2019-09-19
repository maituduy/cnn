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
}