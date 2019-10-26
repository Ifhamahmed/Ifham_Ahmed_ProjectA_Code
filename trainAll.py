from comet_ml import Experiment
# import tensorflow as tf
# import keras.backend.tensorflow_backend as ktf
import os

# def get_session(gpu_fraction=0.90):
#     num_threads = os.environ.get('OMP_NUM_THREADS')
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
#     if num_threads:
#         return tf.Session(config=tf.ConfigProto(gpu_options, intra_op_parallelism_threads=num_threads))
#     else:
#         return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#
# ktf.set_session(get_session())
import matplotlib

experiment = Experiment("m487mKQwNTqFF4z7aZRX3Xv19", project_name="TrainAll_New", log_env_gpu=True)
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

input_size = [256, 256]
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

# build model
model = UNET.build((256, 256, 3), num_of_Classes, pretrained_weights=None)
model.summary()

# learning parameters
INIT_LR = 0.001
EPOCHS = 60
BS = 4

# initialize data generators
traingen = threadsafe_iter(train_set.data_gen(should_augment=True, batch_size=BS))
valgen = threadsafe_iter(val_set.data_gen(should_augment=False, batch_size=BS))

# initialise variables
No_of_train_images = len(os.listdir(dataset_path + '/train_frames/train'))
No_of_val_images = len(os.listdir(dataset_path + '/val_frames/val'))
print("Number of Training Images = {}".format(No_of_train_images))

weights_path = os.getcwd() + '/output/weights.h5'
opt = Adam(lr=INIT_LR, decay=0.0000001)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc', dice_coef, mean_iou, mean_iou_2, jacard_coef])

# early stopping parameters and output
checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
csvlogger = CSVLogger('./log.out', append=True, separator=';')
earlystopping = EarlyStopping(monitor='val_loss', verbose=1, min_delta=0.001, patience=50, mode='min', baseline=0.5)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='min', min_delta=0.0001,
                              cooldown=0, min_lr=0.0000001)

callbacks_list = [checkpoint, csvlogger, earlystopping, reduce_lr]

# train network
print("[INFO] Training network (UNET).........")
H = model.fit_generator(traingen, epochs=EPOCHS,
                        steps_per_epoch=No_of_train_images // BS,
                        validation_data=valgen,
                        validation_steps=No_of_val_images // BS,
                        callbacks=callbacks_list,
                        workers=6)

# evaluate Network
print("[INFO] Evaluating network (UNET).........")

# plot the training loss and accuracy
N = np.arange(0, len(H.history["loss"]))
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss (CE)")
plt.plot(N, H.history["val_loss"], label="val_loss (CE)")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.plot(N, H.history["mean_iou_2"], label="Mean IoU ( Mean Across Class)")
plt.title("Training Loss and Accuracy (UNET)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"] + '/' + 'UNETConvT_SCR_plot.png')

# save the model
print("[INFO] Saving the model (UNET)...")
model.save(args["model"] + '/UNETConvT_SCR.model')
model.save_weights(args["model"] + '/' + 'UNETConvT_SCR_weights.h5')

############################################# FCN - 8 ########################################

model_FCN = FCN.build((256, 256, 3), num_of_Classes, pretrained_weights=None)
model_FCN.summary()

# initialise variables
EPOCHS = 60
INIT_LR = 0.001
weights_path = os.getcwd() + '/output/FCN_weights.h5'
opt = Adam(lr=INIT_LR, decay=0.0000001)

model_FCN.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc', mean_iou, dice_coef, mean_iou_2, jacard_coef])

# early stopping parameters and output
checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
csvlogger = CSVLogger('./logFCN.out', append=True, separator=';')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, mode='min', min_delta=0.0001,
                              cooldown=0, min_lr=0.000001)

callbacks_list = [checkpoint, csvlogger, earlystopping, reduce_lr]

# initialize data generators
traingen = threadsafe_iter(train_set.data_gen(should_augment=True, batch_size=BS))
valgen = threadsafe_iter(val_set.data_gen(should_augment=False, batch_size=BS))

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
plt.savefig(args["plot"] + '/' + 'FCN_Drop_plot.png')

# save the model
print("[INFO] Saving the model (FCN)...")
model_FCN.save(args["model"] + '/FCN_Drop.model')
model_FCN.save_weights(args["model"] + '/' + 'FCN_Drop_weights.h5')

############################################ SegNet #######################################

model_SegNet = SegNet.build((256, 256, 3), num_of_Classes, pretrained_weights=None)
model_SegNet.summary()

# initialise variables
EPOCHS = 60
INIT_LR = 0.001
opt = Adam(lr=INIT_LR, decay=0.0000001)
weights_path = os.getcwd() + '/output/SegNet_weights.h5'
model_SegNet.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc', mean_iou, mean_iou_2, dice_coef, jacard_coef])

# early stopping parameters and output
checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
csvlogger = CSVLogger('./logSegNet.out', append=True, separator=';')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='min', min_delta=0.0001,
                              cooldown=0, min_lr=0.0000001)

callbacks_list = [checkpoint, csvlogger, earlystopping, reduce_lr]

# initialize data generators
traingen = threadsafe_iter(train_set.data_gen(should_augment=True, batch_size=BS))
valgen = threadsafe_iter(val_set.data_gen(should_augment=False, batch_size=BS))

# train network
print("[INFO] Training network (SegNet).........")
H = model_SegNet.fit_generator(traingen,
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
plt.plot(N, H.history["mean_iou_2"], label="Mean IoU(Across Class)")
plt.title("Training Loss and Accuracy (SegNet)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"] + '/' + 'SegNet_SCR_plot.png')

# save the model
print("[INFO] Saving the model (SegNet)...")
model_SegNet.save(args["model"] + '/SegNet_SCR.model')
model_SegNet.save_weights(args["model"] + '/' + 'SegNet_SCR_weights.h5')
print('[INFO] Process Finished!!')
experiment.end()
