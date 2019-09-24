#include "u4.h"

namespace model_zoo {

    U4::U4(Shape input_shape, int start_neural) {
        auto input = new Input(input_shape);
        auto out = U4::build(input, start_neural);
        model = new Model(out);
        model->separate();
    }

    Layer * U4::build(Layer *input, int start_neural) {
        auto conv1 = Conv2d(start_neural * 1, 3, Padding::SAME, 1, Func::NONE)(input);
        conv1 = U4::residual_block(conv1, start_neural * 1);
        conv1 = U4::residual_block(conv1, start_neural * 1, true);
        auto pool1 = Pooling2d(2)(conv1);

        auto conv2 = Conv2d(start_neural * 2, 3, Padding::SAME, 1, Func::NONE)(pool1);
        conv2 = U4::residual_block(conv2, start_neural * 2);
        conv2 = U4::residual_block(conv2, start_neural * 2, true);
        auto pool2 = Pooling2d(2)(conv2);

        auto conv3 = Conv2d(start_neural * 4, 3, Padding::SAME, 1, Func::NONE)(pool2);
        conv3 = U4::residual_block(conv3, start_neural * 4);
        conv3 = U4::residual_block(conv3, start_neural * 4, true);
        auto pool3 = Pooling2d(2)(conv3);

        auto conv4 = Conv2d(start_neural * 8, 3, Padding::SAME, 1, Func::NONE)(pool3);
        conv4 = U4::residual_block(conv4, start_neural * 8);
        conv4 = U4::residual_block(conv4, start_neural * 8, true);
        auto pool4 = Pooling2d(2)(conv4);

        auto convm = Conv2d(start_neural * 16, 3, Padding::SAME, 1, Func::NONE)(pool4);
        convm = U4::residual_block(convm, start_neural * 16);
        convm = U4::residual_block(convm, start_neural * 16, true);

        auto deconv4 = Conv2dTranspose(start_neural * 8, 3, Padding::SAME, 2, Func::NONE)(convm);
        auto uconv4 = Concatenate({deconv4, conv4})(deconv4);

        uconv4 =  Conv2d(start_neural * 8, 3, Padding::SAME, 1, Func::NONE)(uconv4);
        uconv4 = U4::residual_block(uconv4, start_neural * 8);
        uconv4 = U4::residual_block(uconv4, start_neural * 8, true);

        auto deconv3 = Conv2dTranspose(start_neural * 4, 3, Padding::SAME, 2, Func::NONE)(uconv4);
        auto uconv3 = Concatenate({deconv3, conv3})(deconv3);

        uconv3 =  Conv2d(start_neural * 4, 3, Padding::SAME, 1, Func::NONE)(uconv3);
        uconv3 = U4::residual_block(uconv3, start_neural * 4);
        uconv3 = U4::residual_block(uconv3, start_neural * 4, true);

        auto deconv2 = Conv2dTranspose(start_neural * 2, 3, Padding::SAME, 2, Func::NONE)(uconv3);
        auto uconv2 = Concatenate({deconv2, conv2})(deconv2);

        uconv2 =  Conv2d(start_neural * 2, 3, Padding::SAME, 1, Func::NONE)(uconv2);
        uconv2 = U4::residual_block(uconv2, start_neural * 2);
        uconv2 = U4::residual_block(uconv2, start_neural * 2, true);

        auto deconv1 = Conv2dTranspose(start_neural * 1, 3, Padding::SAME, 2, Func::NONE)(uconv2);
        auto uconv1 = Concatenate({deconv1, conv1})(deconv1);

        uconv1 =  Conv2d(start_neural * 1, 3, Padding::SAME, 1, Func::NONE)(uconv1);
        uconv1 = U4::residual_block(uconv1, start_neural * 1);
        uconv1 = U4::residual_block(uconv1, start_neural * 1, true);

        auto output_layer_noActi = Conv2d(1, 1, Padding::SAME, 1, Func::NONE)(uconv1);
        auto output_layer = Activation(Func::SIGMOID)(output_layer_noActi);
        return output_layer;
    }

    Model* U4::get() {
        return model;
    }

    Layer *U4::batch_active(Layer *in) {
        auto x = BatchNormalization()(in);
        x = dynamic_cast<Activation *>(Activation(Func::RELU)(x));
        return x;
    }

    Layer *U4::convolution_block(Layer *in, int n_filters, int kernel_size, int stride, Padding padding, bool activation) {
        auto x = Conv2d(n_filters, kernel_size, padding, stride, Func::NONE)(in);
        if (activation) {
            x = U4::batch_active(x);
        }
        return x;
    }

    Layer *U4::residual_block(Layer *in, int n_filters, int batch_active) {
        auto x = U4::batch_active(in);
        x = U4::convolution_block(x, n_filters, 3);
        x = U4::convolution_block(x, n_filters, 3, 1, Padding::SAME, false);
        x = Add(x,in)(x);
        if (batch_active)
            x = U4::batch_active(x);
        return x;
    }
}