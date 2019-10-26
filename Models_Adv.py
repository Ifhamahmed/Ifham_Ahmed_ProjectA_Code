import numpy as np
import os
# import skimage.io as io
# import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras import backend as K


############################################ UNET ##############################################################

class UNET:
    @staticmethod
    def build(input_shape, classes=2, pretrained_weights=None):
        inputshape = input_shape
        gt_input = Input((input_shape[0], input_shape[1], classes))
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
        drop6 = Dropout(0.5)(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2))(drop6))
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
        conv10 = Conv2D(classes, (1, 1), activation='softmax', name='seg_map')(conv9)

        model = Model(input=[inputs, gt_input], output=[conv10], name='UNET')

        if (pretrained_weights):
            model.load_weights(pretrained_weights)

        return model

######################################### Stanford Adversarial ####################################################


class Stanford_Adversarial:
    @staticmethod
    def build(im_shape, seg_shape):
        im_input = Input(im_shape)
        seg_input = Input(seg_shape)

        chanDim = -1

        # Model Definition
        # Branch 1
        # Conv 1
        branch1 = Conv2D(64, (5, 5), activation='relu', padding='same', kernel_initializer='he_normal',
                         name='branch_1_Conv_1')(seg_input)

        # Branch 2
        # Conv 1
        branch2 = Conv2D(16, (5, 5), activation='relu', padding='same', kernel_initializer='he_normal',
                         name='branch_2_Conv_1')(im_input)
        branch2 = Conv2D(64, (5, 5), activation='relu', padding='same', kernel_initializer='he_normal',
                         name='branch_2_Conv_2')(branch2)

        # Feature Concatenation
        concat = concatenate([branch1, branch2], axis=chanDim)

        # Merged Path
        # Conv-1
        c4 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                    name='merge_Conv_1')(concat)
        pool1 = MaxPooling2D(pool_size=(2, 2))(c4)

        # Conv-ReLu-MaxPool 2
        c5 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                    name='merge_Conv_2')(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(c5)

        # Conv-ReLu 3
        c6 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                    name='merge_Conv_3')(pool2)

        # Conv 4
        c7 = Conv2D(2, (3, 3), activation='sigmoid', padding='same', kernel_initializer='he_normal',
                    name='merge_Conv_4')(c6)
        pad_val = (im_shape[0] - (im_shape[0] // 4)) // 2 # as four sides
        p1 = ZeroPadding2D((pad_val))(c7)

        model = Model(input=[im_input, seg_input], output=p1, name='Stanford_Adversarial')

        return model

############################################ SegNet ################################################

class SegNet:
    @staticmethod
    def build(inputshape, n_labels, pretrained_weights=None):
        input_shape = inputshape
        gt_input = Input((input_shape[0], input_shape[1], n_labels))
        chanDim = -1

        kernel = 3
        output_mode = 'softmax'
        pool_size = (2, 2)

        if K.image_data_format() == 'channels first':
            chanDim = 1
            input_shape = (input_shape[2], input_shape[0], input_shape[1])

        inputs = Input(shape=input_shape)

        conv_1 = Conv2D(64, kernel, padding="same")(inputs)
        conv_1 = BatchNormalization()(conv_1)
        conv_1 = Activation("relu")(conv_1)
        conv_2 = Conv2D(64, kernel, padding="same")(conv_1)
        conv_2 = BatchNormalization()(conv_2)
        conv_2 = Activation("relu")(conv_2)

        pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)

        conv_3 = Conv2D(128, kernel, padding="same")(pool_1)
        conv_3 = BatchNormalization()(conv_3)
        conv_3 = Activation("relu")(conv_3)
        conv_4 = Conv2D(128, kernel, padding="same")(conv_3)
        conv_4 = BatchNormalization()(conv_4)
        conv_4 = Activation("relu")(conv_4)

        pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)

        conv_5 = Conv2D(256, kernel, padding="same")(pool_2)
        conv_5 = BatchNormalization()(conv_5)
        conv_5 = Activation("relu")(conv_5)
        conv_6 = Conv2D(256, kernel, padding="same")(conv_5)
        conv_6 = BatchNormalization()(conv_6)
        conv_6 = Activation("relu")(conv_6)
        conv_7 = Conv2D(256, kernel, padding="same")(conv_6)
        conv_7 = BatchNormalization()(conv_7)
        conv_7 = Activation("relu")(conv_7)

        pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_7)

        conv_8 = Conv2D(512, kernel, padding="same")(pool_3)
        conv_8 = BatchNormalization()(conv_8)
        conv_8 = Activation("relu")(conv_8)
        conv_9 = Conv2D(512, kernel, padding="same")(conv_8)
        conv_9 = BatchNormalization()(conv_9)
        conv_9 = Activation("relu")(conv_9)
        conv_10 = Conv2D(512, kernel, padding="same")(conv_9)
        conv_10 = BatchNormalization()(conv_10)
        conv_10 = Activation("relu")(conv_10)

        pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_10)

        conv_11 = Conv2D(512, kernel, padding="same")(pool_4)
        conv_11 = BatchNormalization()(conv_11)
        conv_11 = Activation("relu")(conv_11)
        conv_12 = Conv2D(512, kernel, padding="same")(conv_11)
        conv_12 = BatchNormalization()(conv_12)
        conv_12 = Activation("relu")(conv_12)
        conv_13 = Conv2D(512, kernel, padding="same")(conv_12)
        conv_13 = BatchNormalization()(conv_13)
        conv_13 = Activation("relu")(conv_13)

        pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(conv_13)

        # decoder

        unpool_1 = MaxUnpooling2D(pool_size)([pool_5, mask_5])

        conv_14 = Conv2D(512, kernel, padding="same")(unpool_1)
        conv_14 = BatchNormalization()(conv_14)
        conv_14 = Activation("relu")(conv_14)
        conv_15 = Conv2D(512, kernel, padding="same")(conv_14)
        conv_15 = BatchNormalization()(conv_15)
        conv_15 = Activation("relu")(conv_15)
        conv_16 = Conv2D(512, kernel, padding="same")(conv_15)
        conv_16 = BatchNormalization()(conv_16)
        conv_16 = Activation("relu")(conv_16)

        unpool_2 = MaxUnpooling2D(pool_size)([conv_16, mask_4])

        conv_17 = Conv2D(512, kernel, padding="same")(unpool_2)
        conv_17 = BatchNormalization()(conv_17)
        conv_17 = Activation("relu")(conv_17)
        conv_18 = Conv2D(512, kernel, padding="same")(conv_17)
        conv_18 = BatchNormalization()(conv_18)
        conv_18 = Activation("relu")(conv_18)
        conv_19 = Conv2D(256, kernel, padding="same")(conv_18)
        conv_19 = BatchNormalization()(conv_19)
        conv_19 = Activation("relu")(conv_19)

        unpool_3 = MaxUnpooling2D(pool_size)([conv_19, mask_3])

        conv_20 = Conv2D(256, kernel, padding="same")(unpool_3)
        conv_20 = BatchNormalization()(conv_20)
        conv_20 = Activation("relu")(conv_20)
        conv_21 = Conv2D(256, kernel, padding="same")(conv_20)
        conv_21 = BatchNormalization()(conv_21)
        conv_21 = Activation("relu")(conv_21)
        conv_22 = Conv2D(128, kernel, padding="same")(conv_21)
        conv_22 = BatchNormalization()(conv_22)
        conv_22 = Activation("relu")(conv_22)

        unpool_4 = MaxUnpooling2D(pool_size)([conv_22, mask_2])

        conv_23 = Conv2D(128, kernel, padding="same")(unpool_4)
        conv_23 = BatchNormalization()(conv_23)
        conv_23 = Activation("relu")(conv_23)
        conv_24 = Conv2D(64, kernel, padding="same")(conv_23)
        conv_24 = BatchNormalization()(conv_24)
        conv_24 = Activation("relu")(conv_24)

        unpool_5 = MaxUnpooling2D(pool_size)([conv_24, mask_1])

        conv_25 = Conv2D(64, kernel, padding="same")(unpool_5)
        conv_25 = BatchNormalization()(conv_25)
        conv_25 = Activation("relu")(conv_25)

        conv_26 = Conv2D(n_labels, (1, 1), padding="valid")(conv_25)
        conv_26 = BatchNormalization()(conv_26)
        outputs = Activation(output_mode)(conv_26)

        model = Model(inputs=[inputs, gt_input], outputs=outputs, name="SegNet")

        if pretrained_weights:
            model.load_weights(pretrained_weights)

        return model


############################################# FCN-8 Segmentor #################################################
class FCN:
    @staticmethod
    def build(inputshape, numOfClasses, pretrained_weights=None):
        weight_path = 'vgg_weights.h5'
        input_shape = inputshape
        gt_input = Input((input_shape[0], input_shape[1], numOfClasses))
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

        model = Model(input=[inputs, gt_input], output=o, name='FCN-8')

        if pretrained_weights:
            model.load_weights(pretrained_weights)

        return model


######################################### Custom Layers ###########################################################
class MaxPoolingWithArgmax2D(Layer):

    def __init__(
            self,
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same',
            **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        if K.backend() == 'tensorflow':
            ksize = [1, pool_size[0], pool_size[1], 1]
            padding = padding.upper()
            strides = [1, strides[0], strides[1], 1]
            output, argmax = K.tf.nn.max_pool_with_argmax(
                inputs,
                ksize=ksize,
                strides=strides,
                padding=padding)
        else:
            errmsg = '{} backend is not supported for layer {}'.format(
                K.backend(), type(self).__name__)
            raise NotImplementedError(errmsg)
        argmax = K.cast(argmax, K.floatx())
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [
            dim // ratio[idx]
            if dim is not None else None
            for idx, dim in enumerate(input_shape)]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]


class MaxUnpooling2D(Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        with K.tf.variable_scope(self.name):
            mask = K.cast(mask, 'int32')
            input_shape = K.tf.shape(updates, out_type='int32')
            #  calculation new shape
            if output_shape is None:
                output_shape = (
                    input_shape[0],
                    input_shape[1] * self.size[0],
                    input_shape[2] * self.size[1],
                    input_shape[3])
            self.output_shape1 = output_shape

            # calculation indices for batch, height, width and feature maps
            one_like_mask = K.ones_like(mask, dtype='int32')
            batch_shape = K.concatenate(
                [[input_shape[0]], [1], [1], [1]],
                axis=0)
            batch_range = K.reshape(
                K.tf.range(output_shape[0], dtype='int32'),
                shape=batch_shape)
            b = one_like_mask * batch_range
            y = mask // (output_shape[2] * output_shape[3])
            x = (mask // output_shape[3]) % output_shape[2]
            feature_range = K.tf.range(output_shape[3], dtype='int32')
            f = one_like_mask * feature_range

            # transpose indices & reshape update values to one dimension
            updates_size = K.tf.size(updates)
            indices = K.transpose(K.reshape(
                K.stack([b, y, x, f]),
                [4, updates_size]))
            values = K.reshape(updates, [updates_size])
            ret = K.tf.scatter_nd(indices, values, output_shape)
            return ret

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return (
            mask_shape[0],
            mask_shape[1] * self.size[0],
            mask_shape[2] * self.size[1],
            mask_shape[3]
        )
