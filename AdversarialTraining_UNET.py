# import packages
from comet_ml import Experiment

experiment = Experiment("m487mKQwNTqFF4z7aZRX3Xv19", project_name="AdversarialTraining_UNET", log_env_gpu=True)

from keras.optimizers import SGD, Adam
from dataLoader import Dataloader
import matplotlib.pyplot as plt
from keras.layers import *
from keras.models import *
import pandas as pd
import numpy as np
import os
import cv2
import pandas
from tqdm import tqdm
from keras import backend as K
from Models import UNET, Stanford_Adversarial


class SS_GAN(object):
    def __init__(self, image_shape, numClasses, pretrained_weights, palette, channels=None):
        """
        :param image_shape: input image shape
        :param numClasses: number of classes in segmentation network
        :param pretrained_weights: pretrained weights for generator model (Not Optional)
        :param palette: colour palette for interpreting probability maps
        :param channels: number of channels in image and ground truth label maps, default=[3, 3]
        """
        if channels is None:
            channels = [3, 3]

        if pretrained_weights is None:
            raise ValueError('The generator model must be pre-trained!!')

        self.numOfClasses = numClasses
        self.pretrained_weights = pretrained_weights
        self.palette = palette

        # Training Parameters for adversarial (From Journal)
        self.INIT_LR = 0.00001
        self.weight = 2

        # Learning parameters for segmentor
        self.INIT_LR_Seg = 0.001

        # model parameters
        self.seg_shape = (image_shape[0], image_shape[1], numClasses)
        self.input_shape = (image_shape[0], image_shape[1], channels[0])

        # optimizers
        opt_dis = Adam(lr=self.INIT_LR)
        opt_gen = Adam(lr=self.INIT_LR_Seg)

        # Build Generator
        self.generator = UNET.build(self.input_shape, self.numOfClasses, pretrained_weights=pretrained_weights)
        self.generator.summary()

        # Build and Compile Discriminator
        self.discriminator = Stanford_Adversarial.build(self.input_shape, self.seg_shape)
        self.discriminator.trainable = True
        self.discriminator.compile(optimizer=opt_dis, loss='binary_crossentropy', metrics=['accuracy'])
        self.discriminator.summary()

        # Define Composite (Segmentor -> Adversarial) Model
        self.composite = self._define_composite(self.INIT_LR_Seg)

        # Compile Segmentor
        self.generator.compile(optimizer=opt_gen, loss='categorical_crossentropy', metrics=['accuracy'])

    def _segmentor_loss(self, y_true, y_pred):  ## Not Used
        """
        :param y_true: Expected(Ground Truth) Confidence Map
        :param y_pred: Predicted Confidence Map
        :return: Second term of segmentation network loss from Journal
        """
        loss = self.weight * K.binary_crossentropy(y_true, y_pred, from_logits=False)
        return K.mean(loss, axis=-1)

    def _adv_loss(self, syn_ohm, true_ohm):
        """
        :param syn_ohm: synthetic one hot map tensor
        :param true_ohm: True input one hot map tensor
        :return: Loss Function Handle
        """
        cce_loss = K.categorical_crossentropy(true_ohm, syn_ohm, from_logits=False)
        weight = self.weight

        def custom_loss(y_true, y_pred):
            """
            :param y_true: True Confidence Maps
            :param y_pred: Synthetic Confidence Maps
            :return: Custom Loss (Segmentor Loss From Journal)
            """
            w_bce_loss = weight * K.mean(K.binary_crossentropy(y_true, y_pred, from_logits=False), axis=-1)
            return cce_loss + w_bce_loss

        return custom_loss

    def _define_composite(self, learning_rate):
        """
        :param learning_rate: initial learning RATE
        :return: composite (Segmentor -> Adversarial) model
        """
        # define the GAN Model
        # RGB image input
        input1 = Input(shape=self.input_shape, name='Image')
        input2 = Input(shape=self.seg_shape, name='SegMap')
        # segmentor output
        output1 = self.generator([input1, input2])

        # concatenate output of segmentor and input RGB image
        concat = [input1, output1]

        # freeze discriminator weights
        self.discriminator.trainable = False

        # input concatenated input(RGB) and output(segmentor) to both inputs of discriminator and obtain output
        output2 = self.discriminator(concat)

        # create composite GAN Model
        model = Model(input=[input1, input2], output=output2, name='Composite')

        # Compile GAN Model with weighted loss function
        opt = Adam(lr=learning_rate)
        model.compile(loss=self._adv_loss(output1, input2), optimizer=opt, metrics=['accuracy'])
        model.summary()

        return model

    # Adversarial training algorithm
    def train_Adversarial(self, dataset, testset, valset, epochs, n_batch, dataset_len):
        """
        :param dataset: Training Dataset
        :param testset: Testing/Plotting Dataset
        :param valset: Validation Dataset
        :param epochs: Number of training epochs
        :param n_batch: Batch Size
        :param dataset_len: Number of Training images (to specify steps per epoch)
        :return: No return value/object

        :info segmentor: Standalone segmentation network
        :info composite: Composite model with the pre-trained segmentation network
                         and the Discriminator with its weights frozen
        :info adversarial: Discriminator/Adversarial Network
        """

        global g_loss, adv_loss, prev_genloss, gen_loss, dis_loss, gen_acc, dis_acc
        prev_genloss = 0
        dis_loss = []
        gen_loss = []
        gen_acc = []
        dis_acc = []
        val_loss = []
        val_acc = []
        steps = 0
        val_steps = []
        steps_per_epoch = dataset_len // n_batch
        output_shape = self.discriminator.output_shape[1:]

        # initialise data generator
        datagenerator = dataset.data_gen(should_augment=True, batch_size=n_batch)
        valgenerator = valset.data_gen(should_augment=False, batch_size=n_batch)
        testgenerator = testset.data_gen(should_augment=False, batch_size=1)

        # define channel 0 (first channel) as synthetic segmentation map channel
        # define channel 1 (second channel)  as ground truth segmentation map channel
        zeros_channel = np.zeros((n_batch, output_shape[0] // 4, output_shape[1] // 4, 1))
        ones_channel = np.ones((n_batch, output_shape[0] // 4, output_shape[1] // 4, 1))
        # pad zeros for loss calculation
        pad_wid = (output_shape[1] - output_shape[1] // 4)
        zeros_channel = np.pad(zeros_channel, mode='constant', pad_width=((0, 0), (0, pad_wid), (0, pad_wid), (0, 0)))
        ones_channel = np.pad(ones_channel, mode='constant', pad_width=((0, 0), (0, pad_wid), (0, pad_wid), (0, 0)))

        # ones_channel = np.ones((n_batch, output_shape[0], output_shape[1], 1))
        # zeros_channel = np.zeros((n_batch, output_shape[0], output_shape[1], 1))

        # define batch wise confidence map for synthetic ohm (one-hot-map)
        y_fake = np.concatenate((ones_channel, zeros_channel), axis=3)

        # define batch wise confidence map for real_ohm
        y_real = np.concatenate((zeros_channel, ones_channel), axis=3)

        # adversarial training of pre-trained Segmentor
        print("[INFO] Commencing Adversarial Training of Segmentor...........")
        for epoch in range(0, epochs):
            print("Epoch {}".format(epoch + 1))
            for _ in tqdm(range(0, steps_per_epoch)):
                # datagenerator output already in batches
                # random batch of real training images and segmentation maps
                real_img, real_ohm = next(datagenerator)

                # generate batch of segmentation maps
                synthetic_ohm = self.generator.predict([real_img, real_ohm])

                ######## Train Adversarial Network ########
                # training discriminator for true segmentation maps
                # print(self.discriminator.layers[-1].get_weights()[-1])
                adv_loss_real = self.discriminator.train_on_batch([real_img, real_ohm], y_real)
                # print(self.discriminator.layers[-1].get_weights()[-1])
                # training discriminator for synthetic segmentation maps
                adv_loss_fake = self.discriminator.train_on_batch([real_img, synthetic_ohm], y_fake)
                # print(self.discriminator.layers[-1].get_weights()[-1])
                # average loss for discriminator/adversarial
                adv_loss = np.add(adv_loss_fake, adv_loss_real)
                dis_loss.append(adv_loss[0])
                dis_acc.append(0.5 * adv_loss[1])

                ############# Train Segmentation Network ###############
                # generate a new batch of images and segmentation maps
                real_img, real_ohm = next(datagenerator)
                # update weights of segmentor/composite model according to loss function(LF) in journal
                # print(self.generator.layers[-1].get_weights()[-1])
                g_loss = self.composite.train_on_batch([real_img, real_ohm], y_real)
                # print(self.generator.layers[-2].get_weights()[-1])
                # average generator loss
                gen_loss.append(g_loss[0])
                gen_acc.append(g_loss[1])
                # gen_lone_loss.append(g_cce_loss[0])

                steps += 1
                # Plot The Progress
                print(
                    "{} [D loss: {}, acc.: {}%] [Composite G loss: {}]".format(epoch, adv_loss[0], 50 * adv_loss[1],
                                                                               g_loss[0]))
            # save image every 5 epochs
            if (epoch+1) % 5 == 0:
                print("[INFO] Saving a test image................")
                self._sample_image(epoch=epoch, testgenerator=testgenerator)

            # Model Checkpoint
            self._model_checkpoint(g_loss=g_loss[0], prev_g_loss=prev_genloss)
            prev_genloss = g_loss[0]

            # Evaluate on validation data after every epoch
            vloss, vacc = self._evaluate_on_valset(valgenerator, epoch)
            val_loss.append(vloss)
            val_acc.append(vacc)

            # count steps per validation
            val_steps.append(steps_per_epoch * (epoch + 1))

        # end of training algorithm
        print("[INFO] Evaluating Model..............")
        self._plot_results_and_save(g_loss=gen_loss, g_acc=gen_acc, d_loss=dis_loss, d_acc=dis_acc, steps=steps,
                                    val_steps=val_steps,
                                    val_loss=val_loss,
                                    val_acc=val_acc)
        print("[INFO] Saving Model Weights..........")
        self._save_weights()
        print("[INFO] Saving Data...................")
        data = pd.DataFrame(data={"g_loss": gen_loss, "g_acc": gen_acc, "d_loss": dis_loss, "d_acc":dis_acc,
                                    "steps": np.arange(steps).tolist()})
        data.to_csv("./output/training_data.csv", sep=',', index=False)
        data = pd.DataFrame(data={"val_steps": val_steps, "val_loss": val_loss, "val_acc": val_acc})
        data.to_csv("./output/validation_data.csv", sep=',', index=False)
        print("[INFO] Process Finished..............")

    def _sample_image(self, epoch, testgenerator):
        """
        :param epoch: epoch number
        :param testgenerator: test data generator for producing images
        :return: no return value, saves one image per preset number of epochs in output folder
        """
        # inference test image every k epochs
        test_img, test_ohm = next(testgenerator)

        # start of debugging Statement
        # image = np.zeros((1, 256, 256, 3)).astype("float")
        # test_img = cv2.imread(
        #     '/home/ifham_fyp/PycharmProjects/dataset/test_frames/test/aachen_000001_000019_leftImg8bit.png')
        # test_img = cv2.resize(test_img, (256, 256))
        # image[0] = test_img
        # # end of debugging statement

        preds = self.generator.predict([test_img, test_ohm])

        # remove batch dimension
        preds = preds.reshape((preds.shape[1], preds.shape[2], preds.shape[3]))
        preds = preds.argmax(axis=-1)

        # collapse class probabilities to label map
        label_map = np.zeros((preds.shape[0], preds.shape[1], 3)).astype('float32')
        for ind in range(0, len(self.palette)):
            submat = np.where(preds == ind)

            np.put(label_map[:, :, 0], np.ravel_multi_index(np.array(submat), label_map.shape[0:2]),
                   self.palette[ind][0])
            np.put(label_map[:, :, 1], np.ravel_multi_index(np.array(submat), label_map.shape[0:2]),
                   self.palette[ind][1])
            np.put(label_map[:, :, 2], np.ravel_multi_index(np.array(submat), label_map.shape[0:2]),
                   self.palette[ind][2])

        epoch_num = str(epoch)
        im = cv2.resize(test_img[0], (self.input_shape[0], self.input_shape[1]))
        cv2.imwrite('output' + '/' + epoch_num + 'img.png', im * 255)
        cv2.imwrite('output' + '/' + epoch_num + '.png', label_map)

    def _save_weights(self):
        """
        :info: Saves Model Weights after training
        """
        self.generator.save_weights('output' + '/AdvUNET.h5')

    def _evaluate_on_valset(self, val_set, epoch):
        """
        :param val_set: Validation Data Generator
        :param epoch: Epoch number
        :return:
        """
        test_img, test_ohm = next(val_set)
        eval_loss = self.generator.test_on_batch([test_img, test_ohm], test_ohm)
        print("{} [Generator Validation: loss: {}, acc.: {}%]".format(epoch, eval_loss[0], 100 * eval_loss[1]))
        return eval_loss[0], eval_loss[1]

    def _plot_results_and_save(self, g_loss, g_acc, d_loss, d_acc, steps, val_steps, val_loss, val_acc):
        """
        :param g_loss: Gen Loss
        :param g_acc: Gen Accuracy
        :param d_loss: Discriminator Loss
        :param d_acc: Discriminator Accuracy
        :param val_loss: Validation Loss on a batch
        :param val_acc: Validation Accuracy on a batch
        :param val_steps: Total number of training iterations per validation iteration
        :param steps: Training iterations carried out
        :info: Plots the results and saves them into the output folder
        """
        n = np.arange(0, steps)
        plt.style.use('ggplot')
        plt.figure()

        plt.plot(n, g_loss, label="Average Generator (Composite) Loss")
        plt.plot(n, g_acc, label="Generator Accuracy")
        plt.plot(n, d_loss, label="Average Discrminator Loss")
        plt.plot(n, d_acc, label="Discriminator Accuracy")
        plt.plot(val_steps, val_loss, label="G Validation Loss")
        plt.plot(val_steps, val_acc, label="G Validation Accuracy")

        plt.title("Training/Val Loss and Accuracy (Semantic Segmentation-Adversarial)")
        plt.xlabel("Steps #")
        plt.ylabel("Loss/Accuracy")

        plt.legend()
        plt.savefig('output/' + 'AdvUNET_plot.png')

    def _model_checkpoint(self, g_loss, prev_g_loss):
        """
        :param g_loss: Current Generator Loss
        :param prev_g_loss: Previous Generator Loss
        :info: Compares the losses. Saves model weights into the output folder
        """
        if (prev_g_loss - g_loss) > 0.1:
            self.generator.save_weights(filepath='output/checkpoint_weights.h5', overwrite=True)


if __name__ == '__main__':
    # import colour palette
    df = pd.read_csv('classes.csv', ",", header=None)
    palette = np.array(df.values, dtype=np.uint8)
    num_of_Classes = palette.shape[0]

    # make output folder
    if 'output' not in os.listdir(os.getcwd()):
        os.makedirs(os.getcwd() + '/output')

    # Dataset path
    dataset_path = '/home/ifham_fyp/PycharmProjects/dataset'

    train_frame_path = os.path.sep.join([dataset_path, "train_frames/train"])
    train_mask_path = os.path.sep.join([dataset_path, "train_masks/train"])

    val_frame_path = os.path.sep.join([dataset_path, "val_frames/val"])
    val_mask_path = os.path.sep.join([dataset_path, "val_masks/val"])

    test_frame_path = os.path.sep.join([dataset_path, "test_frames/test"])
    test_mask_path = os.path.sep.join([dataset_path, "test_masks/test"])

    # initialise variables
    No_of_train_images = len(os.listdir(dataset_path + '/train_frames/train'))
    No_of_val_images = len(os.listdir(dataset_path + '/val_frames/val'))
    print("Number of Training Images = {}".format(No_of_train_images))

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

    test_set = Dataloader(image_paths=test_frame_path,
                          mask_paths=test_mask_path,
                          image_size=input_size,
                          numclasses=num_of_Classes,
                          channels=[3, 3],
                          palette=palette,
                          seed=47)

    # instantiate SS_GAN Class
    Semantic_GAN = SS_GAN(image_shape=(256, 256),
                          numClasses=num_of_Classes,
                          pretrained_weights='UNET_weights.h5',
                          palette=palette,
                          channels=[3, 3])

    # Training Parameters
    EPOCHS = 60
    BS = 8

    # Train Segmentation model using the adversarial network
    Semantic_GAN.train_Adversarial(train_set, test_set, val_set, epochs=EPOCHS, n_batch=BS,
                                   dataset_len=No_of_train_images)
    experiment.end()
