from keras import backend as K
import tensorflow as tf
import numpy as np


def iou_coef(self, y_true, y_pred):
    smooth = 1e-6
    intersection = K.cast(K.all(K.stack([y_true, y_pred], axis=0), axis=0), dtype='float32')
    union = K.cast(K.any(K.stack([y_true, y_pred], axis=0), axis=0), dtype='float32')
    iou = K.sum(intersection, axis=[1, 2]) / K.sum((union + smooth), axis=[1, 2])

    return K.mean(iou, axis=[-1])


def mean_iou(y_true, y_pred):
    prec = []
    NUM_CLASSES = 29
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, NUM_CLASSES)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def mean_iou_2(y_true, y_pred, NUM_CLASSES=29):
    score, up_opt = tf.metrics.mean_iou(y_true, y_pred, NUM_CLASSES)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score


def dice_coef(y_true, y_pred):
    smooth = 1e-6
    intersection = K.sum(y_true * y_pred, axis=[1, 2])
    union = K.sum(y_true * y_true, axis=[1, 2]) + K.sum(y_pred * y_pred, axis=[1, 2])
    dice = K.mean((2. * intersection) / (union + smooth), axis=[-1])
    return dice


def dice_coef_loss(self, y_true, y_pred):
    return 1 - self.dice_coef(y_true, y_pred)


def iou_coef_loss(y_true, y_pred):
    return 1 - iou_coef(y_true, y_pred)


def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)



# a = K.variable([[1, 0, 1], [1, 1, 1]])
# b = K.variable([[0, 1, 1], [0, 1, 0]])
#
# c = K.stack([a, b], axis=0)
# c = K.all(c, axis=0)
# c = K.cast(c, dtype='float32')
# print(K.eval(c))
