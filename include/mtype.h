#pragma once
#include <variant>

namespace mtype {
    template<typename T>
    std::ostream& operator<<(typename std::enable_if<std::is_enum<T>::value, std::ostream>::type& stream, const T& e) {
        return stream << static_cast<typename std::underlying_type<T>::type>(e);
    }
    enum class Position{
        LEFT,
        RIGHT,
        TOP,
        BOTTOM
    };
    enum class Padding{
        SAME,
        VALID
    };

    enum class PoolingMode {
        MAX,
        AVERAGE,
        AVERAGE_TF
    };

    enum class Func{
        SIGMOID,
        RELU,
        NONE
    };

    
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
                return out << "(" << shape.batch << ", " << shape.w << ", " << shape.h << ", " << shape.c << ")";
            }
        
    };

    typedef std::map<std::string, std::variant<int, std::string, Shape, Padding, PoolingMode, Shape*>> Dict;


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