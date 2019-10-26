import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from Models import UNET, SegNet, FCN
import numpy as np
import pandas as pd
from keras.layers import *
import os
import cv2
import cupy as cp
from dataLoader import Dataloader, threadsafe_iter
from tqdm import tqdm
import matlab.engine
import matlab
from bfmeasure import bfscore
eng = matlab.engine.start_matlab()

# class names of 29 unique classes
names = ['unlabeled', 'building', 'sky', 'vegetation', 'pole', 'traffic sign', 'traffic light', 'bus', 'car',
        'person', 'road', 'sidewalk', 'terrain', 'dynamic', 'bicycle', 'ground', 'wall', 'fence', 'rider',
        'motorcycle', 'bridge', 'parking', 'truck', 'train', 'caravan', 'trailer', 'guard rail', 'rail track', 'tunnel']

############################################## Evaluation #################################################
class evaluation(object):
    def __init__(self, image_shape, numClasses, palette,  pretrained_unet=None, pretrained_fcn=None, pretrained_segnet=None,
                 unet=False, fcn=False, segnet=False, channels=None):
        if channels is None:
            channels = [3, 3]

        # model selection
        self.unetchoice = unet
        self.fcnchoice = fcn
        self.segnetchoice = segnet

        # model parameters
        self.input_shape = (image_shape[0], image_shape[1], channels[0])
        self.num_of_classes = numClasses
        self.palette = palette
        if unet:
            if not pretrained_unet:
                raise ValueError("Need Weights to Evaluate Model")
            else:
                self.unet = UNET.build(self.input_shape, self.num_of_classes, pretrained_weights=pretrained_unet)

        if fcn:
            if not pretrained_fcn:
                raise ValueError("Need Weights to Evaluate Model")
            else:
                self.fcn = FCN.build(self.input_shape, self.num_of_classes, pretrained_weights=pretrained_fcn)

        if segnet:
            if not pretrained_segnet:
                raise ValueError("Need Weights to Evaluate Model")
            else:
                self.segnet = SegNet.build(self.input_shape, self.num_of_classes, pretrained_weights=pretrained_segnet)

    def _compute_giou(self, y_pred, y_true, smooth=0.001):
        """
        :param y_pred: Predicted Segementation Maps
        :param y_true: Ground Truth Segmentation Maps
        :param smooth: Smoothing Factor to avoid divison by zero
        :return: Global IoU
        """
        """
        INFO:
        only evaluate for 19 classes
        computes Global IoU over Dataset
        disregards classes in a particular image where union is zero (class doesnt exist in ground truth image)
        Accumulates true positives over class in all images
        Accumulates False positives + True Positives + False Negatives over class in all images
        """
        weights = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0])
        y_pred = y_pred[:, :, :, np.nonzero(weights)[0]]
        y_true = y_true[:, :, :, np.nonzero(weights)[0]]

        #########################################################

        num_classes = y_pred.shape[-1]
        y_pred_ohm = (np.array([y_pred.argmax(axis=-1) == s for s in range(num_classes)]) * 1).transpose(1, 2, 3, 0)
        intersection = np.sum(np.logical_and(y_pred_ohm, y_true), axis=(1, 2))
        union = np.sum(np.logical_or(y_pred_ohm, y_true), axis=(1, 2))

        union_non_zero = 0
        intersection_non_zero = 0
        for i in range(union.shape[1]):
            indices = np.nonzero(union[:, i])[0]
            union_non_zero += np.sum(union[indices, i])
            intersection_non_zero += np.sum(intersection[indices, i])


        iou = (intersection_non_zero + smooth) / (union_non_zero + smooth)
        return iou

    def _compute_miou(self, y_pred, y_true, smooth=0.001):
        """
        :param y_pred: Predicted Segementation Maps
        :param y_true: Ground Truth Segmentation Maps
        :param smooth: Smoothing Factor to avoid divison by zero
        :return: mIoU averaged over class
        """
        """
        INFO:
        only evaluate for 11 classes
        computes mIoU over Dataset
        Accumulates true positives over class in all images
        Accumulates False positives + True Positives + False Negatives over class in all images
        Obtained IoU per class for each class then averages by number of evaluated classes
        """
        #weights = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0])
        weights = np.array([0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        y_pred = y_pred[:, :, :, np.nonzero(weights)[0]]
        y_true = y_true[:, :, :, np.nonzero(weights)[0]]

        class_names = [names[i] for i in np.nonzero(weights)[0]]

        num_classes = y_pred.shape[-1]
        y_pred_ohm = (np.array([y_pred.argmax(axis=-1) == s for s in range(num_classes)]) * 1).transpose(1, 2, 3, 0)
        intersection = np.sum(np.logical_and(y_pred_ohm, y_true), axis=(1, 2))
        union = np.sum(np.logical_or(y_pred_ohm, y_true), axis=(1, 2))

        iou = 0
        for i in tqdm(range(y_true.shape[-1])):
            indices = np.nonzero(union[:, i])[0]
            union_non_zero = np.sum(union[indices, i])
            intersection_non_zero = np.sum(intersection[indices, i])
            iou_c = (intersection_non_zero + smooth) / (union_non_zero + smooth)
            iou += iou_c
            print(iou_c * 100, class_names[i])

        return iou / num_classes



    def _compute_mean_pixel_accuracy(self, y_pred, y_true):
        """
        :param y_pred: Predicted Segementation Maps
        :param y_true: Ground Truth Segmentation Maps
        :return: Global mPA
        """
        """
        only evaluate for 11 classes
        Finds pixel accuracy per class over all images 
        """
        #weights = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0])
        weights = np.array([0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        y_pred = y_pred[:, :, :, np.nonzero(weights)[0]]
        y_true = y_true[:, :, :, np.nonzero(weights)[0]]

        num_classes = y_pred.shape[-1]
        y_pred_ohm = (np.array([y_pred.argmax(axis=-1) == s for s in range(num_classes)]) * 1).transpose(1, 2, 3, 0)

        common_pixels = np.sum(np.logical_and(y_pred_ohm, y_true), axis=(1, 2))
        total_pixels_per_class = np.sum(y_true, axis=(0, 1, 2))
        mPA = 0
        for i in range(common_pixels.shape[1]):
            common_pixels_per_class = np.sum(common_pixels[:, i])
            mPA += common_pixels_per_class / total_pixels_per_class[i]

        return mPA / num_classes

    def _compute_BF_measure(self, y_pred, y_true):
        """
        :param y_pred: Predicted Segementation Maps
        :param y_true: Ground Truth Segmentation Maps
        :return: mBF measure averaged over all images
        """
        BF = np.zeros((y_pred.shape[0]))
        #weights = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0])
        weights = np.array([0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        for i in tqdm(range(0, y_pred.shape[0])):
            prediction = y_pred[i].argmax(axis=-1)
            gt = y_true[i].argmax(axis=-1)

            # collapse class probabilities to label map
            label_map = np.zeros((prediction.shape[0], prediction.shape[1], 3)).astype('float32')
            gt_label_map = np.zeros((prediction.shape[0], prediction.shape[1], 3)).astype('float32')
            for ind in range(0, len(self.palette)):
                submat = np.where(prediction == ind)

                np.put(label_map[:, :, 0], np.ravel_multi_index(np.array(submat), label_map.shape[0:2]),
                       self.palette[ind][0])
                np.put(label_map[:, :, 1], np.ravel_multi_index(np.array(submat), label_map.shape[0:2]),
                       self.palette[ind][1])
                np.put(label_map[:, :, 2], np.ravel_multi_index(np.array(submat), label_map.shape[0:2]),
                       self.palette[ind][2])

                submat = np.where(gt == ind)
                np.put(gt_label_map[:, :, 0], np.ravel_multi_index(np.array(submat), gt_label_map.shape[0:2]),
                       self.palette[ind][0])
                np.put(gt_label_map[:, :, 1], np.ravel_multi_index(np.array(submat), gt_label_map.shape[0:2]),
                       self.palette[ind][1])
                np.put(gt_label_map[:, :, 2], np.ravel_multi_index(np.array(submat), gt_label_map.shape[0:2]),
                       self.palette[ind][2])

            BF[i] = bfscore(gt_label_map, label_map, self.palette, weights)

        return 100 * np.mean(BF)

    def _get_unet_predictions(self, testgenerator, dataset_len, batchsize=4):
        steps_per_epoch = dataset_len // batchsize
        predictions = np.zeros((dataset_len, self.input_shape[0], self.input_shape[1], self.num_of_classes))
        groundtruth = np.zeros((dataset_len, self.input_shape[0], self.input_shape[1], self.num_of_classes))

        c = 0
        print('[INFO] Getting UNET Predictions..............')
        for _ in tqdm(range(steps_per_epoch)):
            img, gt = next(testgenerator)
            preds = self.unet.predict_on_batch(img)

            predictions[c, :, :, :] = preds
            groundtruth[c, :, :, :] = gt
            c += 1

        return predictions, groundtruth

    def _get_fcn_predictions(self, testgenerator, dataset_len, batchsize=4):
        steps_per_epoch = dataset_len // batchsize
        predictions = np.zeros((dataset_len, self.input_shape[0], self.input_shape[1], self.num_of_classes))
        groundtruth = np.zeros((dataset_len, self.input_shape[0], self.input_shape[1], self.num_of_classes))

        c = 0
        print('[INFO] Getting FCN Predictions..............')
        for _ in tqdm(range(steps_per_epoch)):
            img, gt = next(testgenerator)
            preds = self.fcn.predict_on_batch(img)

            predictions[c, :, :, :] = preds
            groundtruth[c, :, :, :] = gt
            c += 1

        return predictions, groundtruth

    def _get_segnet_predictions(self, testgenerator, dataset_len, batchsize=4):
        steps_per_epoch = dataset_len // batchsize
        predictions = np.zeros((dataset_len, self.input_shape[0], self.input_shape[1], self.num_of_classes))
        groundtruth = np.zeros((dataset_len, self.input_shape[0], self.input_shape[1], self.num_of_classes))

        c = 0
        print('[INFO] Getting SegNet Predictions..............')
        for _ in tqdm(range(steps_per_epoch)):
            img, gt = next(testgenerator)
            preds = self.segnet.predict_on_batch(img)

            predictions[c, :, :, :] = preds
            groundtruth[c, :, :, :] = gt
            c += 1

        return predictions, groundtruth

    def evaluate(self, testgenerator, dataset_len, batchsize=4):
        global mIoU_unet, mIoU_fcn, mIoU_segnet

        if self.unetchoice:
            preds_unet, gt_unet = self._get_unet_predictions(testgenerator=testgenerator, dataset_len=dataset_len,
                                                             batchsize=batchsize)
            print('[INFO] Evaluating UNET Predictions..............')
            gIoU_unet = self._compute_giou(preds_unet, gt_unet)
            print('Global Intersection Over Union for the UNET on Test Dataset: ' + str(gIoU_unet * 100))
            mIoU_unet = self._compute_miou(preds_unet, gt_unet)
            print('Mean Intersection Over Union for the UNET on Test Dataset: ' + str(mIoU_unet * 100))
            bf_unet = self._compute_BF_measure(preds_unet, gt_unet)
            print('Mean BF Measure for the UNET on Test Dataset: ' + str(bf_unet))
            mPA_unet = self._compute_mean_pixel_accuracy(preds_unet, gt_unet)
            print('Mean Pixel Accuracy Measure for the UNET on Test Dataset: ' + str(mPA_unet * 100))

        if self.fcnchoice:
            preds_fcn, gt_fcn = self._get_fcn_predictions(testgenerator=testgenerator, dataset_len=dataset_len,
                                                          batchsize=batchsize)
            print('[INFO] Evaluating FCN Predictions..............')
            gIoU_fcn = self._compute_giou(preds_fcn, gt_fcn)
            print('Global Intersection Over Union for the FCN on Test Dataset: ' + str(gIoU_fcn * 100))
            mIoU_fcn = self._compute_miou(preds_fcn, gt_fcn)
            print('Mean Intersection Over Union for the FCN on Test Dataset: ' + str(mIoU_fcn * 100))
            bf_fcn = self._compute_BF_measure(preds_fcn, gt_fcn)
            print('Mean BF Measure for the FCN on Test Dataset: ' + str(bf_fcn))
            mPA_fcn = self._compute_mean_pixel_accuracy(preds_fcn, gt_fcn)
            print('Mean Pixel Accuracy Measure for the FCN on Test Dataset: ' + str(mPA_fcn * 100))

        if self.segnetchoice:
            preds_segnet, gt_segnet = self._get_segnet_predictions(testgenerator=testgenerator, dataset_len=dataset_len,
                                                                   batchsize=batchsize)
            print('[INFO] Evaluating SegNet Predictions..............')
            gIoU_segnet = self._compute_giou(preds_segnet, gt_segnet)
            print('Global Intersection Over Union for the SegNet on Test Dataset: ' + str(gIoU_segnet * 100))
            mIoU_segnet = self._compute_miou(preds_segnet, gt_segnet)
            print('Mean Intersection Over Union for the SegNet on Test Dataset: ' + str(mIoU_segnet * 100))
            bf_segnet = self._compute_BF_measure(preds_segnet, gt_segnet)
            print('Mean BF Measure for the SegNet on Test Dataset: ' + str(bf_segnet))
            mPA_segnet = self._compute_mean_pixel_accuracy(preds_segnet, gt_segnet)
            print('Mean Pixel Accuracy Measure for the SegNet on Test Dataset: ' + str(mPA_segnet * 100))

        print('[INFO] Evaluation Done..............')




if __name__ == '__main__':
    # Get path of test images and masks
    test_datapath = '/home/ifham_fyp/PycharmProjects/dataset'
    test_imagepath = os.path.sep.join([test_datapath, "test_frames/test"])
    test_maskpath = os.path.sep.join([test_datapath, "test_masks/test"])

    # input size
    input_size = [256, 256]

    # import colour palette
    df = pd.read_csv('classes.csv', ",", header=None)
    palette = np.array(df.values, dtype=np.uint8)
    num_of_Classes = palette.shape[0]

    # instantiate datagenerator class
    test_set = Dataloader(image_paths=test_imagepath,
                          mask_paths=test_maskpath,
                          image_size=input_size,
                          numclasses=num_of_Classes,
                          channels=[3, 3],
                          palette=palette,
                          seed=47)

    # evaluation parameters
    BS = 1

    # initialize data generators with threading lock
    testgen = threadsafe_iter(test_set.data_gen(should_augment=False, batch_size=BS))

    No_of_test_images = len(os.listdir(test_imagepath))
    print("Number of Test Images = {}".format(No_of_test_images))

    eval = evaluation(input_size,
                      num_of_Classes,
                      palette,
                      pretrained_unet='models/AdvUNET.h5',
                      pretrained_fcn='models/AdvFCN.h5',
                      pretrained_segnet='models/AdvSegnet.h5',
                      unet=True,
                      fcn=True,
                      segnet=True)

    eval.evaluate(testgenerator=testgen, dataset_len=No_of_test_images, batchsize=BS)

