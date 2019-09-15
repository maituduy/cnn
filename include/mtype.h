#pragma once
#include <variant>
#include "enum.h"

namespace mtype {
    
    BETTER_ENUM(Position, int, LEFT, RIGHT, TOP, BOTTOM)
    BETTER_ENUM(Padding, int, SAME, VALID)
    BETTER_ENUM(PoolingMode, int, MAX, AVERAGE, AVERAGE_TF)

    struct Shape {
        unsigned int batch,w,h,c;
        
        Shape(unsigned int _batch, unsigned int _w, unsigned int _h, unsigned int _c) {
            batch = _batch;
            w = _w;
            h = _h;
            c = _c;
        }

        public:
            friend std::ostream& operator<<(std::ostream& out, const Shape& shape){
                return out << "Shape(" << shape.batch << ", " << shape.w << ", " << shape.h << ", " << shape.c << ")";
            }
        
    };

    typedef std::map<std::string, std::variant<int, std::string, Shape, Padding, PoolingMode>> Dict;


    struct Size {
        unsigned int w,h;

        Size(unsigned int _w, unsigned int _h) {
            w = _w;
            h = _h;
        }
    };
    
    struct PaddingShape {
        unsigned int t,b,l,r;

        PaddingShape(unsigned int _t, unsigned int _b, unsigned int _l, unsigned int _r) {
            t = _t;
            b = _b;
            l = _l;
            r = _r;
        }
    };
    
}