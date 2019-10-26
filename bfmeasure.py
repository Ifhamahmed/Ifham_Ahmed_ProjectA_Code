# -*- coding:utf-8 -*-

# bfscore: Contour/Boundary matching score for multi-class image segmentation #
# Reference: Csurka, G., D. Larlus, and F. Perronnin. "What is a good evaluation measure for semantic segmentation?" Proceedings of the British Machine Vision Conference, 2013, pp. 32.1-32.11. #
# Crosscheck: https://www.mathworks.com/help/images/ref/bfscore.html #
# Obtained from https://github.com/minar09/bfscore_python/blob/master/bfscore.py
# Had to adapt to my models and code structure. There were many errors and version differences.

import cv2
import numpy as np
import math

major = cv2.__version__.split('.')[0]  # Get opencv version
bDebug = False

""" For precision, contours_a==GT & contours_b==Prediction
    For recall, contours_a==Prediction & contours_b==GT """


def calc_precision_recall(contours_a, contours_b, threshold):
    top_count = 0

    try:
        for b in range(len(contours_b)):

            # find the nearest distance
            for a in range(len(contours_a)):
                dist = (contours_a[a][0] - contours_b[b][0]) * \
                       (contours_a[a][0] - contours_b[b][0])
                dist = dist + \
                       (contours_a[a][1] - contours_b[b][1]) * \
                       (contours_a[a][1] - contours_b[b][1])
                if dist < threshold * threshold:
                    top_count = top_count + 1
                    break

        precision_recall = top_count / len(contours_b)
    except Exception as e:
        precision_recall = 0

    return precision_recall, top_count, len(contours_b)


""" computes the BF (Boundary F1) contour matching score between the predicted and GT segmentation """


def bfscore(gtfile, prfile, palette, weights, threshold=5):
    gt__ = gtfile # cv2.imread(gtfile)  # Read GT segmentation
    # cv2.imshow('gt', gt__/255)
    # cv2.imshow('pr', prfile / 255)
    gt_ = cv2.cvtColor(gt__, cv2.COLOR_BGR2GRAY)  # Convert color space

    pr__ = prfile # cv2.imread(prfile)  # Read predicted segmentation
    pr_ = cv2.cvtColor(pr__, cv2.COLOR_BGR2GRAY)  # Convert color space

    # only evaluate for selected 19 classes
    eval_classes = palette[palette.any(axis=1) == weights]
    classes_gt = np.unique(gt__.reshape(-1, gt__.shape[2]), axis=0).astype('uint8') # Get GT classes
    classes_pr = np.unique(pr__.reshape(-1, pr__.shape[2]), axis=0).astype('uint8') # Get predicted classes

    # Check classes from GT and prediction
    if not np.array_equiv(classes_gt, classes_pr):
        # print('Classes are not same! GT:', classes_gt, 'Pred:', classes_pr)

        classes = np.concatenate((classes_gt, classes_pr), axis=0)
        classes = np.unique(classes, axis=0)
        indices = np.array(np.all((eval_classes[:, None, :] == classes[None, :, :]), axis=-1).nonzero()).T
        classes = palette[indices[:, 0]]
        # print('Merged classes :', classes)
    else:
        # print('Classes :', classes_gt)
        classes = classes_gt  # Get matched classes

    m = classes.shape[0]  # Get max of classes (number of classes)
    # print(m)
    # Define bfscore variable (initialized with zeros)
    bfscores = np.zeros((m + 1), dtype=float)
    areas_gt = np.zeros((m + 1), dtype=float)

    for i in range(m + 1):
        bfscores[i] = 0
        areas_gt[i] = 0
    # print(classes.shape[0])
    for target_class in range(classes.shape[0]):  # Iterate over classes

        if classes[target_class].any() == 0:  # Skip background / Unlabelled Regions
            continue

        # print(">>> Calculate for class:", classes_gt[target_class])

        gt = gt_.copy()
        # print(classes_gt[target_class])
        indices = np.where((gt__ != classes[target_class]).all(axis=2))
        #print(indices)
        gt[indices[0], indices[1]] = 0
        gt = gt.astype('uint8')
        # print(gt.shape)

        # contours point list
        if major == '3':  # For opencv version 3.x
            _, contours, _ = cv2.findContours(
                gt, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # Find contours of the shape
        else:  # For other opencv versions
            contours, _ = cv2.findContours(
                gt, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # Find contours of the shape
            # print(contours)

        # contours list of numpy arrays
        contours_gt = []
        for i in range(len(contours)):
            for j in range(len(contours[i])):
                contours_gt.append(contours[i][j][0].tolist())
        if bDebug:
            print('contours_gt')
            print(contours_gt)

        # Get contour area of GT
        if contours_gt:
            area = cv2.contourArea(np.array(contours_gt))
            areas_gt[target_class] = area
        #print(areas_gt[0])

        # print("\tArea:", areas_gt[target_class])

        # Draw GT contours
        img = np.zeros_like(gt__)
        # print(img.shape)
        indices = np.where((gt__ == classes[target_class]).any(axis=2))
        #print(indices)
        img[indices[0], indices[1], :] = 128  # Blue
        img = cv2.drawContours(img, contours, -1, (255, 0, 0), 1)

        pr = pr_.copy()
        indices = np.where((pr__ != classes[target_class]).all(axis=2))
        #print(indices)
        pr[indices[0], indices[1]] = 0
        pr = pr.astype('uint8')
        # print(pr.shape)

        # contours point list
        if major == '3':  # For opencv version 3.x
            _, contours, _ = cv2.findContours(
                pr, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        else:  # For other opencv versions
            contours, _ = cv2.findContours(
                pr, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # contours list of numpy arrays
        contours_pr = []
        for i in range(len(contours)):
            for j in range(len(contours[i])):
                contours_pr.append(contours[i][j][0].tolist())

        if bDebug:
            print('contours_pr')
            print(contours_pr)

        # Draw predicted contours
        indices = np.where((pr__ == classes[target_class]).any(axis=2))
        #print(indices)
        img[indices[0], indices[1], :] = 128  # Red
        img = cv2.drawContours(img, contours, -1, (0, 0, 255), 1)

        # 3. calculate
        precision, numerator, denominator = calc_precision_recall(
            contours_gt, contours_pr, threshold)  # Precision
        # print("\tprecision:", denominator, numerator)

        recall, numerator, denominator = calc_precision_recall(
            contours_pr, contours_gt, threshold)  # Recall
        # print("\trecall:", denominator, numerator)

        f1 = 0
        try:
            f1 = 2 * recall * precision / (recall + precision)  # F1 score
        except:
            # f1 = 0
            f1 = 0 # np.nan
        # print("\tf1:", f1)
        bfscores[target_class] = f1

    #     cv2.imshow('image', img)
    #     cv2.waitKey(0)
    #
    # cv2.destroyAllWindows()

    # return bfscores[1:], np.sum(bfscores[1:])/len(classes[1:])    # Return bfscores, except for background, and per image score
    return np.mean(bfscores[1:-1])# Return bfscores


if __name__ == "__main__":

    sample_gt = 'data/gt_1.png'
    # sample_gt = 'data/gt_0.png'

    sample_pred = 'data/crf_1.png'
    # sample_pred = 'data/pred_0.png'

    score, areas_gt = bfscore(sample_gt, sample_pred, 2)  # Same classes
    # score, areas_gt = bfscore(sample_gt, sample_pred, 2)    # Different classes

    # gt_shape = cv2.imread('data/gt_1.png').shape
    # print("Total area:", gt_shape[0] * gt_shape[1])

    total_area = np.nansum(areas_gt)
    print("GT area (except background):", total_area)
    fw_bfscore = []
    for each in zip(score, areas_gt):
        if math.isnan(each[0]) or math.isnan(each[1]):
            fw_bfscore.append(math.nan)
        else:
            fw_bfscore.append(each[0] * each[1])
    print(fw_bfscore)

    print("\n>>>>BFscore:\n")
    print("BFSCORE:", score)
    print("Per image BFscore:", np.nanmean(score))

    print("\n>>>>Weighted BFscore:\n")
    print("Weighted-BFSCORE:", fw_bfscore)
    print("Per image Weighted-BFscore:", np.nansum(fw_bfscore) / total_area)
