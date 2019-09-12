#pragma once

namespace mtype {
    
    enum Position {
        LEFT,
        RIGHT,
        TOP,
        BOTTOM
    };

    enum Padding {
        SAME,
        VALID
    };

    enum PoolingMode {
        MAX,
        AVERAGE,
        AVERAGE_TF
    };

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