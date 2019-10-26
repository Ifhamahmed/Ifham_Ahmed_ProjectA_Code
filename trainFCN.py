from comet_ml import Experiment
import os
import matplotlib

experiment = Experiment("m487mKQwNTqFF4z7aZRX3Xv19", project_name="Train_FCN", log_env_gpu=True)
matplotlib.use("Agg")

# import packages
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
# from imutils import paths
from keras.optimizers import Adam, SGD
from dataLoader import Dataloader, threadsafe_iter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
from Models import UNET, FCN, SegNet
from LossFunc import dice_coef, mean_iou_2, mean_iou, jacard_coef

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Input Path of dataset")
ap.add_argument("-m", "--model", required=True, help="Path of location to save model")
ap.add_argument("-p", "--plot", required=True, help="Path of location to save plot")
args = vars(ap.parse_args())

# get dataset path
dataset_path = args["dataset"]

# import colour palette
df = pd.read_csv('classes.csv', ",", header=None)
palette = np.array(df.values, dtype=np.uint8)
num_of_Classes = palette.shape[0]

# initialize data and label paths
print("[INFO] Creating Datagenerator Objects........")

train_frame_path = os.path.sep.join([args["dataset"], "train_frames/train"])
train_mask_path = os.path.sep.join([args["dataset"], "train_masks/train"])

val_frame_path = os.path.sep.join([args["dataset"], "val_frames/val"])
val_mask_path = os.path.sep.join([args["dataset"], "val_masks/val"])

input_size = [352, 352]
# instantiate datagenerator class
train_set = Dataloader(image_paths=train_frame_path,
                       mask_paths=train_mask_path,
                       image_size=input_size,
                       numclasses=num_of_Classes,
                       channels=[3, 3],
                       palette=palette,
                       seed=47)

val_set = Dataloader(image_paths=val_frame_path,
                     mask_paths=val_mask_path,
                     image_size=input_size,
                     numclasses=num_of_Classes,
                     channels=[3, 3],
                     palette=palette,
                     seed=47)

# Build Model
model_FCN = FCN.build((352, 352, 3), num_of_Classes, pretrained_weights="FCN_weights.h5")
model_FCN.summary()

# learning parameters
INIT_LR = 0.001
EPOCHS = 100
BS = 2

# initialize data generators
traingen = threadsafe_iter(train_set.data_gen(should_augment=True, batch_size=BS))
valgen = threadsafe_iter(val_set.data_gen(should_augment=False, batch_size=BS))

# Print INFO
No_of_train_images = len(os.listdir(dataset_path + '/train_frames/train'))
No_of_val_images = len(os.listdir(dataset_path + '/val_frames/val'))
print("Number of Training Images = {}".format(No_of_train_images))

# optimizer
opt = Adam(lr=INIT_LR, decay=0.0000001)

# early stopping parameters and Saving Training Data
weights_path = os.getcwd() + '/output/weights.h5'
checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
csvlogger = CSVLogger('./log_FCN.out', append=True, separator=';')
earlystopping = EarlyStopping(monitor='val_loss', verbose=1, min_delta=0.001, patience=50, mode='min', baseline=0.5)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='min', min_delta=0.0001,
                              cooldown=0, min_lr=0.0000001)

# Model Compilation
callbacks_list = [checkpoint, csvlogger, earlystopping, reduce_lr]
model_FCN.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc', mean_iou, dice_coef, mean_iou_2, jacard_coef])

# train network
print("[INFO] Training network (FCN).........")
H = model_FCN.fit_generator(traingen,
                            epochs=EPOCHS,
                            steps_per_epoch=No_of_train_images // BS,
                            validation_steps=No_of_val_images // BS,
                            validation_data=valgen,
                            callbacks=callbacks_list,
                            workers=6)

# plot the training loss and accuracy
N = np.arange(0, len(H.history["loss"]))
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss (CE)")
plt.plot(N, H.history["val_loss"], label="val_loss (CE)")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.plot(N, H.history["mean_iou_2"], label="Mean IoU( Mean Across Class)")
plt.title("Training Loss and Accuracy (FCN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"] + '/' + 'FCN_plot.png')

# save the model
print("[INFO] Saving the model (FCN)...")
model_FCN.save(args["model"] + '/FCN.model')
model_FCN.save_weights(args["model"] + '/' + 'FCN_weights.h5')
print('[INFO] Process Finished!!')
experiment.end()
