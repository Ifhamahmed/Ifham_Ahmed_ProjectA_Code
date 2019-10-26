import random
import threading
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import cupy as cp
from PIL import Image
from skimage.transform import rescale, rotate
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator

class threadsafe_iter:
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)


class Dataloader(object):
    ######################## Initialize Class Instantiation ########################
    def __init__(self, image_paths, mask_paths, image_size, numclasses, channels=None, palette=None, seed=None):
        if channels is None:
            channels = [3, 12]
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size
        self.channels = channels
        self.palette = palette
        self.numClasses = numclasses
        self.gen = ImageDataGenerator()

        if (seed == None):
            self.seed = random.randint(1, 1000)
        else:
            self.seed = seed
    ############################### Data Augmentations ############################
    def _apply_rotation(self, image, mask, seed):
        random.seed(seed)
        deg = random.randint(0, 30)
        # rot_img = self.gen.apply_transform(x=image, transform_parameters={'theta': deg})
        # rot_mask = self.gen.apply_transform(x=mask, transform_parameters={'theta': deg})

        rot_img = rotate(image=image, angle=deg, mode='reflect', preserve_range=True)
        rot_mask = rotate(image=mask, angle=deg, mode='reflect', order=0, preserve_range=True)

        return rot_img, rot_mask

    def _apply_flip_lr(self, image, mask, seed):
        random.seed(seed)
        cond = random.random()
        if cond > 0.5:
            bl = True
        else:
            bl = False

        flip_img = self.gen.apply_transform(x=image, transform_parameters={'flip_horizontal': bl})
        flip_mask = self.gen.apply_transform(x=mask, transform_parameters={'flip_horizontal': bl})

        return flip_img, flip_mask

    def _corrupt_brightness(self, image, mask, seed):
        random.seed(seed)
        br = random.random()
        corr_img = self.gen.apply_transform(x=image, transform_parameters={'brightness': br})

        return corr_img, mask

    def _rescale_img_mask_in(self, image, mask, seed):
        random.seed(seed)
        sf = round(random.uniform(0.3, 1), 1)
        pad_wid = round(image.shape[0] - sf * image.shape[0])
        pad_hgt = round(image.shape[1] - sf * image.shape[1])


        res_image = rescale(image=image, scale=sf, anti_aliasing=False, mode='reflect', multichannel=True,
                            preserve_range=True)
        res_mask = rescale(image=mask, scale=sf, order=0, anti_aliasing=False, mode='reflect', multichannel=True,
                           preserve_range=True)
        res_image = np.pad(res_image, mode='reflect', pad_width=((0, pad_wid), (0, pad_hgt), (0, 0)))
        res_mask = np.pad(res_mask, mode='reflect', pad_width=((0, pad_wid), (0, pad_hgt), (0, 0)))

        return res_image, res_mask

    def _rescale_img_mask_out(self, image, mask, seed):
        random.seed(seed)
        sf = round(random.uniform(1, 2), 1)

        res_image = rescale(image=image, scale=sf, anti_aliasing=False, mode='reflect', multichannel=True,
                            preserve_range=True)
        res_mask = rescale(image=mask, scale=sf, order=0, anti_aliasing=False, mode='reflect', multichannel=True,
                           preserve_range=True)

        res_image = res_image[0:image.shape[0], 0:image.shape[1], :]
        res_mask = res_mask[0:mask.shape[0], 0:mask.shape[1], :]
        return res_image, res_mask
    ############################## One Hot Encoding of the Ground Truth Label Maps ###########################
    def _one_hot_encode(self, image, mask):
        one_hot_map = []
        for color in self.palette:
            class_map = (color == mask)
            class_map = np.all(class_map, axis=2) * 255
            class_map = class_map.astype('float32')
            one_hot_map.append(class_map)

        return image, one_hot_map
    ################################### Image and Mask Obtaining, Augmenting Function ##################################
    def _get_image_nd_mask(self, index1, index2, should_augment=False):
        image_path = self.image_paths
        mask_path = self.mask_paths
        size = self.image_size
        seed = self.seed

        img = cv2.imread(image_path + '/' + index1)
        img = cv2.resize(img, (size[0], size[1]))
        o_img = img

        mask = cv2.imread(mask_path + '/' + index2)
        mask = cv2.resize(mask, (size[0], size[1]), interpolation=cv2.INTER_NEAREST)
        o_mask = mask

        shd_flip = random.randint(0, 1)
        shd_rot = random.randint(0, 1)
        shd_br = random.randint(0, 1)
        shd_sc = random.randint(0, 1)

        if should_augment:
            if shd_flip:
                img, mask = self._apply_flip_lr(img, mask, seed)
            if shd_rot:
                img, mask = self._apply_rotation(img, mask, seed)
            if shd_br:
                img, mask = self._corrupt_brightness(img, mask, seed)
            if shd_sc:
                img, mask = self._rescale_img_mask_in(img, mask, seed)

        return img, mask#, o_img, o_mask
    ###################################### Data Generator Function #####################################
    def data_gen(self, should_augment=False, batch_size=4):
        # import data paths and size
        image_path = self.image_paths
        mask_path = self.mask_paths
        sz = self.image_size

        if self.palette is None:
            raise ValueError("No palette for one hot encoding is specified!")

        # generate augmented data
        c = 0
        n = sorted(os.listdir(image_path))
        m = sorted(os.listdir(mask_path))
        while True:
            n, m = shuffle(n, m)
            image = np.zeros((batch_size, sz[0], sz[1], 3)).astype("float")
            ohm = np.zeros((batch_size, sz[0], sz[1], self.numClasses)).astype("float32")

            for i in range(c, c + batch_size):
                aug_img, aug_mask = self._get_image_nd_mask(n[i], m[i], should_augment=should_augment)
                image[i - c] = aug_img / 255

                aug_img, aug_ohm = self._one_hot_encode(aug_img, aug_mask)

                train_mask = np.zeros((sz[0], sz[1], self.numClasses)).astype("float32")

                j = 0
                for k in aug_ohm:

                    train_mask[:, :, j] = k / 255
                    j += 1

                ohm[i - c] = train_mask

            c += batch_size
            if c + batch_size >= len(os.listdir(image_path)):
                c = 0
            yield image, ohm#, mask, o_image, o_mask

# df = pd.read_csv('classes.csv', ",", header=None)
# palette = np.array(df.values, dtype=np.uint8)
#
# data = Dataloader(image_paths='dataset/train_frames/train',
#                   mask_paths='dataset/train_masks/train',
#                   image_size=[512, 512],
#                   numclasses=29,
#                   channels=[3, 3],
#                   palette=palette,
#                   seed=47)
#
# d = data.data_gen(True, 4)
# d = threadsafe_iter(d)
# img, ohm, mask, o_image, o_mask = next(d)
#
# mk = ohm[0][:, :, 0]
# mk = np.reshape(mk, (mk.shape[0], mk.shape[1], 1))
# im = img[0]
# cv2.imshow('im', im)
# cv2.imshow('mk', mk)
# cv2.imshow('mask', mask[0])
# cv2.imshow('o_img', o_image[0])
# cv2.imshow('o_mask', o_mask[0])
# cv2.waitKey(0)
#
#
# preds = ohm[0].argmax(axis=-1)
#
# label_map = np.zeros((preds.shape[0], preds.shape[1], 3)).astype('float32')
#
# for ind in range(0, len(palette)):
#     submat = np.where(preds == ind)
#
#     np.put(label_map[:, :, 0], np.ravel_multi_index(np.array(submat), label_map.shape[0:2]), palette[ind][0])
#     np.put(label_map[:, :, 1], np.ravel_multi_index(np.array(submat), label_map.shape[0:2]), palette[ind][1])
#     np.put(label_map[:, :, 2], np.ravel_multi_index(np.array(submat), label_map.shape[0:2]), palette[ind][2])
#
# label_map = label_map / 255
# #cv2.imshow('mask', label_map)
# cv2.waitKey(0)

# print(ohm[0][0].shape)
# plt.imshow(ohm[0][0], cmap=plt.cm.gray)  # use appropriate colormap here
# plt.show()
# arr2d = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint32)
#
# for i in range(0, mask.shape[0]):
#     for j in range(0, mask.shape[1]):
#         key = (mask[i, j, 0], mask[i, j, 1], mask[i, j, 2])
#         tmp = int((np.where(np.all(self.palette == key, axis=1, keepdims=True) == True))[0])
#         if tmp == None:
#             tmp = 5
#         arr2d[i, j] = tmp + 1
# one_hot_map = []
# for i, value in enumerate(self.palette):
#     tmp = np.ones((mask.shape[0], mask.shape[1]), dtype=np.uint8) * (i + 1)
#     classmap = (tmp == arr2d) * (i + 1)
#     one_hot_map.append(np.array(classmap, dtype=np.float32))


# images = []
# one_hot_map = []
# for k in range(0, len(imgs)):
#     tmp_img, tmp_ohm = self._one_hot_encode(imgs[k], mks[k])
#     images.append(tmp_img)
#     one_hot_map.append(tmp_ohm)
