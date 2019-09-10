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
        AVERAGE
    };

    struct Size {
        unsigned int w,h;

        Size(unsigned int _w, unsigned int _h) {
            w = _w;
            h = _h;
        }
    };
    
}