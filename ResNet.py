# import the necessary packages
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import add
from keras.regularizers import l2
from keras import backend as K


class ResNet:
    @staticmethod
    def residual_module(data, filter_num, stride, chan_dim, red=False, reg=0.01, bnEps=2e-5, bnMom=0.9):
        # the shortcut branch of the ResNet module should be
        # initialize as the input (identity) data
        shortcut = data

        # the first block of the ResNet module
        conv1 = Conv2D(filter_num, (3, 3), strides=stride, padding="same", use_bias=False, kernel_regularizer=l2(reg))(data)
        bn1 = BatchNormalization(axis=chan_dim, epsilon=bnEps, momentum=bnMom)(conv1)
        act1 = Activation("relu")(bn1)

        # the second block of the ResNet module
        conv2 = Conv2D(filter_num, (3, 3), strides=stride, padding="same", use_bias=False, kernel_regularizer=l2(reg))(act1)
        bn2 = BatchNormalization(axis=chan_dim, epsilon=bnEps, momentum=bnMom)(conv2)

        # if we are to reduce the spatial size, apply a CONV layer to the shortcut
        if red:
            shortcut = Conv2D(filter_num, (1, 1), strides=stride, padding="same", use_bias=False, kernel_regularizer=l2(reg))(act1)

        # add together the shortcut and the final BN
        x = add([bn2, shortcut])

        act2 = Activation("relu")(x)

        # return the activation as the output of the ResNet module
        return act2

    @staticmethod
    def build(height, width, depth, num_filters, policy_output_dim, reg=0.01, bnEps=2e-5, bnMom=0.9, num_res_blocks=19):
        # initialize the input shape to be "channels last" and the
        # channels dimension itself
        inputShape = (height, width, depth)
        chan_dim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # set the input and apply BN
        inputs = Input(shape=inputShape)

        # apply CONV => BN => ACT
        x = Conv2D(num_filters, (1, 1), use_bias=False, padding="same", kernel_regularizer=l2(reg))(inputs)
        x = BatchNormalization(axis=chan_dim, epsilon=bnEps, momentum=bnMom)(x)
        x = Activation("relu")(x)

        # adding residual modules
        for _ in range(num_res_blocks):
            x = ResNet.residual_module(x, num_filters, (1, 1), chan_dim, reg=reg, bnEps=bnEps, bnMom=bnMom)

        # Policy head
        x_pol = Conv2D(2, (3, 3), strides=(1, 1), padding="same", use_bias=False, kernel_regularizer=l2(reg))(x)
        x_pol = BatchNormalization(axis=chan_dim, epsilon=bnEps, momentum=bnMom)(x_pol)
        x_pol = Activation("relu")(x_pol)
        x_pol = Flatten()(x_pol)
        x_pol = Dense(policy_output_dim, activation="softmax")(x_pol)

        # Value head
        x_val = Conv2D(1, (1, 1), strides=(1, 1), padding="same", use_bias=False, kernel_regularizer=l2(reg))(x)
        x_val = BatchNormalization(axis=chan_dim, epsilon=bnEps, momentum=bnMom)(x_val)
        x_val = Activation("relu")(x_val)
        x_val = Flatten()(x_val)
        x_val = Dense(256, activation="relu", kernel_regularizer=l2(reg))(x_val)
        # x_val = Flatten()(x_val)
        x_val = Dense(1, activation="tanh")(x_val)

        # Create the model
        model = Model(inputs, [x_pol, x_val], name="resnet")

        # return the constructed network architecture
        return model
