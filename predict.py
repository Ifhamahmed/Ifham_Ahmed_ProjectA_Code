from keras.models import load_model
import cv2
import numpy as np
import argparse
import pandas as pd
from Models import UNET, FCN, SegNet
#from LossFunc import dice_coef, mean_iou, mean_iou_2

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="Path to model")
ap.add_argument("-i", "--image", required=True, help="Path to image")
ap.add_argument("-w", "--width", type=int, default=64, help="Width of image")
ap.add_argument("-e", "--height", type=int, default=64, help="height of image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
output = image.copy()

# import colour palette
df = pd.read_csv('classes.csv', ",", header=None)
palette = np.array(df.values, dtype=np.float32)

image = cv2.resize(image, (args["height"], args["width"]))
image = image.astype("float32") / 255

# add batch dimension
image = image.reshape((1, image.shape[0], image.shape[1], 3))

# load model
print("[INFO] Loading Model............")
model = UNET.build((512, 512, 3), 29, pretrained_weights='AdvUNET.h5')
# model = load_model(args["model"], custom_objects={'dice_coef': dice_coef, 'mean_iou': mean_iou,
#                                                   'mean_iou_2': mean_iou_2},)

# make a prediction on the image
preds = model.predict(image)

# collapse class probabilities to label map
preds = preds.reshape((preds.shape[1], preds.shape[2], preds.shape[3]))
preds = preds.argmax(axis=-1)

label_map = np.zeros((preds.shape[0], preds.shape[1], 3)).astype('float32')

for ind in range(0, len(palette)):
    submat = np.where(preds == ind)

    np.put(label_map[:, :, 0], np.ravel_multi_index(np.array(submat), label_map.shape[0:2]), palette[ind][0])
    np.put(label_map[:, :, 1], np.ravel_multi_index(np.array(submat), label_map.shape[0:2]), palette[ind][1])
    np.put(label_map[:, :, 2], np.ravel_multi_index(np.array(submat), label_map.shape[0:2]), palette[ind][2])

label_map = label_map / 255
cv2.imshow('mask', label_map)
cv2.imwrite('AdvUNET.png', label_map*255)
cv2.waitKey(0)











