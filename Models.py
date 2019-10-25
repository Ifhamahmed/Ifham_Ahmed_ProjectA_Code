import numpy as np
import os
# import skimage.io as io
# import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras import backend as K
from keras.regularizers import *

from layers import *


############################################ UNET ##############################################################

class UNET:
    @staticmethod
    def build(input_shape, classes=2, pretrained_weights=None):
        inputshape = input_shape
        chanDim = -1

        if K.image_data_format() == "channels first":
            inputshape = (input_shape[2], input_shape[0], input_shape[1])
            chanDim = 1

        inputs = Input(inputshape)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            Conv2DTranspose(512, kernel_size=(2, 2), strides=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=chanDim)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=chanDim)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        drop7 = Dropout(0.5)(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            Conv2DTranspose(128, kernel_size=(2, 2), strides=(2, 2))(drop7))
        merge8 = concatenate([conv2, up8], axis=chanDim)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=chanDim)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(classes, (1, 1), activation='softmax')(conv9)

        model = Model(input=inputs, output=conv10, name='UNET')

        if (pretrained_weights):
            model.load_weights(pretrained_weights)

        return model


############################################ SegNet ################################################

class SegNet:
    @staticmethod
    def build(inputshape, n_labels, pretrained_weights=None):
        input_shape = inputshape
        chanDim = -1

        kernel = 3
        output_mode = 'softmax'
        pool_size = (2, 2)

        if K.image_data_format() == 'channels first':
            chanDim = 1
            input_shape = (input_shape[2], input_shape[0], input_shape[1])

        inputs = Input(shape=input_shape)

        conv_1 = Convolution2D(64, (kernel, kernel), padding="same")(inputs)
        conv_1 = BatchNormalization()(conv_1)
        conv_1 = Activation("relu")(conv_1)
        conv_2 = Convolution2D(64, (kernel, kernel), padding="same")(conv_1)
        conv_2 = BatchNormalization()(conv_2)
        conv_2 = Activation("relu")(conv_2)

        pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)

        conv_3 = Convolution2D(128, (kernel, kernel), padding="same")(pool_1)
        conv_3 = BatchNormalization()(conv_3)
        conv_3 = Activation("relu")(conv_3)
        conv_4 = Convolution2D(128, (kernel, kernel), padding="same")(conv_3)
        conv_4 = BatchNormalization()(conv_4)
        conv_4 = Activation("relu")(conv_4)

        pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)

        conv_5 = Convolution2D(256, (kernel, kernel), padding="same")(pool_2)
        conv_5 = BatchNormalization()(conv_5)
        conv_5 = Activation("relu")(conv_5)
        conv_6 = Convolution2D(256, (kernel, kernel), padding="same")(conv_5)
        conv_6 = BatchNormalization()(conv_6)
        conv_6 = Activation("relu")(conv_6)
        conv_7 = Convolution2D(256, (kernel, kernel), padding="same")(conv_6)
        conv_7 = BatchNormalization()(conv_7)
        conv_7 = Activation("relu")(conv_7)

        pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_7)

        conv_8 = Convolution2D(512, (kernel, kernel), padding="same")(pool_3)
        conv_8 = BatchNormalization()(conv_8)
        conv_8 = Activation("relu")(conv_8)
        conv_9 = Convolution2D(512, (kernel, kernel), padding="same")(conv_8)
        conv_9 = BatchNormalization()(conv_9)
        conv_9 = Activation("relu")(conv_9)
        conv_10 = Convolution2D(512, (kernel, kernel), padding="same")(conv_9)
        conv_10 = BatchNormalization()(conv_10)
        conv_10 = Activation("relu")(conv_10)

        pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_10)

        conv_11 = Convolution2D(512, (kernel, kernel), padding="same")(pool_4)
        conv_11 = BatchNormalization()(conv_11)
        conv_11 = Activation("relu")(conv_11)
        conv_12 = Convolution2D(512, (kernel, kernel), padding="same")(conv_11)
        conv_12 = BatchNormalization()(conv_12)
        conv_12 = Activation("relu")(conv_12)
        conv_13 = Convolution2D(512, (kernel, kernel), padding="same")(conv_12)
        conv_13 = BatchNormalization()(conv_13)
        conv_13 = Activation("relu")(conv_13)
        drop13 = Dropout(0.5)(conv_13)

        pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(drop13)

        # decoder

        unpool_1 = MaxUnpooling2D(pool_size)([pool_5, mask_5])

        conv_14 = Convolution2D(512, (kernel, kernel), padding="same")(unpool_1)
        conv_14 = BatchNormalization()(conv_14)
        conv_14 = Activation("relu")(conv_14)
        drop14 = Dropout(0.5)(conv_14)
        conv_15 = Convolution2D(512, (kernel, kernel), padding="same")(drop14)
        conv_15 = BatchNormalization()(conv_15)
        conv_15 = Activation("relu")(conv_15)
        conv_16 = Convolution2D(512, (kernel, kernel), padding="same")(conv_15)
        conv_16 = BatchNormalization()(conv_16)
        conv_16 = Activation("relu")(conv_16)

        unpool_2 = MaxUnpooling2D(pool_size)([conv_16, mask_4])

        conv_17 = Convolution2D(512, (kernel, kernel), padding="same")(unpool_2)
        conv_17 = BatchNormalization()(conv_17)
        conv_17 = Activation("relu")(conv_17)
        conv_18 = Convolution2D(512, (kernel, kernel), padding="same")(conv_17)
        conv_18 = BatchNormalization()(conv_18)
        conv_18 = Activation("relu")(conv_18)
        conv_19 = Convolution2D(256, (kernel, kernel), padding="same")(conv_18)
        conv_19 = BatchNormalization()(conv_19)
        conv_19 = Activation("relu")(conv_19)

        unpool_3 = MaxUnpooling2D(pool_size)([conv_19, mask_3])

        conv_20 = Convolution2D(256, (kernel, kernel), padding="same")(unpool_3)
        conv_20 = BatchNormalization()(conv_20)
        conv_20 = Activation("relu")(conv_20)
        conv_21 = Convolution2D(256, (kernel, kernel), padding="same")(conv_20)
        conv_21 = BatchNormalization()(conv_21)
        conv_21 = Activation("relu")(conv_21)
        conv_22 = Convolution2D(128, (kernel, kernel), padding="same")(conv_21)
        conv_22 = BatchNormalization()(conv_22)
        conv_22 = Activation("relu")(conv_22)

        unpool_4 = MaxUnpooling2D(pool_size)([conv_22, mask_2])

        conv_23 = Convolution2D(128, (kernel, kernel), padding="same")(unpool_4)
        conv_23 = BatchNormalization()(conv_23)
        conv_23 = Activation("relu")(conv_23)
        conv_24 = Convolution2D(64, (kernel, kernel), padding="same")(conv_23)
        conv_24 = BatchNormalization()(conv_24)
        conv_24 = Activation("relu")(conv_24)

        unpool_5 = MaxUnpooling2D(pool_size)([conv_24, mask_1])

        conv_25 = Convolution2D(64, (kernel, kernel), padding="same")(unpool_5)
        conv_25 = BatchNormalization()(conv_25)
        conv_25 = Activation("relu")(conv_25)

        conv_26 = Convolution2D(n_labels, (1, 1), padding="valid")(conv_25)
        conv_26 = BatchNormalization()(conv_26)
        outputs = Activation(output_mode)(conv_26)

        model = Model(inputs=inputs, outputs=outputs, name="SegNet")

        if pretrained_weights:
            model.load_weights(pretrained_weights)

        return model


############################################# FCN Segmentor #################################################
class FCN:
    @staticmethod
    def build(inputshape, numOfClasses, pretrained_weights=None):
        weight_path = 'vgg_weights.h5'
        input_shape = inputshape
        IMAGE_ORDERING = "channels_last"

        if K.image_data_format() == 'channels first':
            input_shape = (input_shape[2], input_shape[0], input_shape[1])
            IMAGE_ORDERING = "channels_first"

        inputs = Input(shape=input_shape)

        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING)(
            inputs)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING)(x)
        f1 = x

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING)(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING)(x)
        f2 = x

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING)(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING)(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING)(x)
        pool3 = x

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING)(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING)(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING)(x)
        pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING)(
            x)  # (None, 14, 14, 512)

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING)(
            pool4)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING)(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING)(x)
        pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING)(
            x)  # (None, 7, 7, 512)

        vgg = Model(inputs, pool5)
        vgg.load_weights(weight_path)

        n = 4096
        o = (Conv2D(n, (7, 7), activation='relu', padding='same', name="conv6", data_format=IMAGE_ORDERING))(pool5)
        o = Dropout(0.5)(o)
        conv7 = (Conv2D(n, (1, 1), activation='relu', padding='same', name="conv7", data_format=IMAGE_ORDERING))(o)

        # 4 times upsamping for pool4 layer
        conv7_4 = Conv2DTranspose(numOfClasses, kernel_size=(4, 4), strides=(4, 4), use_bias=False,
                                  data_format=IMAGE_ORDERING)(conv7)
        # (None, 224, 224, 10)
        # 2 times upsampling for pool411
        pool411 = (
            Conv2D(numOfClasses, (1, 1), activation='relu', padding='same', name="pool4_11",
                   data_format=IMAGE_ORDERING))(
            pool4)
        pool411_2 = (
            Conv2DTranspose(numOfClasses, kernel_size=(2, 2), strides=(2, 2), use_bias=False,
                            data_format=IMAGE_ORDERING))(
            pool411)

        pool311 = (
            Conv2D(numOfClasses, (1, 1), activation='relu', padding='same', name="pool3_11",
                   data_format=IMAGE_ORDERING))(
            pool3)

        o = Add(name="add")([pool411_2, pool311, conv7_4])
        o = Conv2DTranspose(numOfClasses, kernel_size=(8, 8), strides=(8, 8), use_bias=False,
                            data_format=IMAGE_ORDERING)(o)
        o = (Activation('softmax'))(o)

        model = Model(inputs, o)

        if pretrained_weights:
            model.load_weights(pretrained_weights)

        return model

