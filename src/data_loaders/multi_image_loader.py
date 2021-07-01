import os
import pandas as pd
import cv2
import glob
import random
import time
from os.path import join

import numpy as np
from keras.applications import imagenet_utils
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence
from src.utils import const,util
from imgaug import augmenters as iaa
def preprocess_input(x, **kwargs):
    """Preprocesses a numpy array encoding a batch of images.

    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].

    # Returns
        Preprocessed array.
    """
    return imagenet_utils.preprocess_input(x, mode='tf',data_format='channels_last', **kwargs)


no_augments = {
    # "rescale":1./255
    "preprocessing_function": preprocess_input
}

# imgaug augmentations

def train_crop(target_size):
    return iaa.Sequential([
            iaa.CropToFixedSize(width=target_size[0], height=target_size[1], position="uniform"),
        ])

def train_augment_iic(target_size):
    return iaa.Sequential([
            iaa.CropToFixedSize(width=target_size[0], height=target_size[1], position="uniform"),
            iaa.Fliplr(0.5),
            iaa.AddToHue((-30,30)), # insspired by 255*0.125
            iaa.AddToSaturation((-100,100)), # inspired by 255*0.4
            iaa.GammaContrast(gamma=(0.6,1.4))
])

def train_augment_affine(target_size):
    return iaa.Sequential([
            iaa.CropToFixedSize(width=target_size[0], height=target_size[1], position="uniform"),
            iaa.Fliplr(0.5),
            iaa.MultiplyHue(mul=(1-0.125,1.125)),
            iaa.MultiplySaturation(mul=(0.6,1.4)),
            iaa.Multiply(mul=(0.6,1.4)),
            iaa.GammaContrast(gamma=(0.6,1.4)),
            iaa.Affine(
                rotate=(-20, 20),  # in degrees
                shear=(-10, 10),  # in degreees
                order=1,  # use bilinear interpolation (fast)
                cval=0,  # if mode is constant, use cval of 0
                mode='constant'  # constant mode
        )
])

def train_augment_affine_cutout(target_size):
    return iaa.Sequential([
            iaa.CropToFixedSize(width=target_size[0], height=target_size[1], position="uniform"),
            iaa.Fliplr(0.5),
            iaa.MultiplyHue(mul=(1-0.125,1.125)),
            iaa.MultiplySaturation(mul=(0.6,1.4)),
            iaa.Multiply(mul=(0.6,1.4)),
            iaa.GammaContrast(gamma=(0.6,1.4)),
            iaa.Cutout(nb_iterations=1,fill_mode="constant",cval=0, size=(0.2,0.7)),
            iaa.Affine(
                rotate=(-20, 20),  # in degrees
                shear=(-10, 10),  # in degreees
                order=1,  # use bilinear interpolation (fast)
                cval=0,  # if mode is constant, use cval of 0
                mode='constant'  # constant mode
        )
])


def val_crop(target_size):
    return iaa.Sequential([
            iaa.CropToFixedSize(width=target_size[0], height=target_size[1], position="center"),
        ])

def function_for_augment_name(name):
    if 'val_crop' == name:
        return val_crop
    elif'train_crop' == name:
        return train_crop
    elif 'train_augment_iic' == name:
        return train_augment_iic
    elif 'train_augment_affine' == name:
        return train_augment_affine
    elif 'train_augment_affine_cutout' == name:
        return train_augment_affine_cutout
    else:
        return None

class ImageLoader(Sequence):
    """
    Generator yields two images, first one is an augment base image,
    second one can be same image (augmented) or a different image from the same or another class
    """

    def __init__(self,name,  directory, batch_size, loading_size, target_size, imgaug_augments=None, shuffle=False, seed_number = None, sample_repetition=1, overcluster_subheads_number=1, gt_subheads_number=1, overcluster_num_classes=20, num_classes=20, unlabeled_data_dir=None, alternate_count = -1, alternate_overcluster_max = 5, alternate_gt_max = 5, adaptive_bs=False, max_percent_unlabeled_data_in_batch=0.5):
        # store paramters
        self.directory = directory
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.target_size = target_size
        self.loading_size = loading_size
        self.sample_repetition = sample_repetition
        self.overcluster_subheads_number = overcluster_subheads_number
        self.gt_subheads_number = gt_subheads_number
        self.overcluster_num_classes = overcluster_num_classes
        self.num_classes = num_classes
        self.subheads_number = overcluster_subheads_number + gt_subheads_number

        # init random
        self.random = random.Random()
        seed = random.randint(0,99999) if seed_number is None else seed_number
        print("%s - Doubleimage loader with seed %d" % (name,seed))
        self.random.seed(seed)

        # file names
        self.file_names_with_labels = glob.glob(directory + "/**/*.png") # get file names
        self.file_names_with_labels = sorted(self.file_names_with_labels)

        if unlabeled_data_dir is not None:
            self.file_names_without_lables = glob.glob(unlabeled_data_dir + "/**/*.png") # get file names
        else:
            self.file_names_without_lables = []
        self.file_names_without_lables = sorted(self.file_names_without_lables)

        # unlabeled data to normal data
        self.file_names = np.array(self.file_names_with_labels + self.file_names_without_lables)


        # get adaptive bs
        if adaptive_bs:
            _ , self.batch_size = util.get_bs_adaptive(len(self.file_names), self.batch_size)
            print("chose adaptive bs of ", self.batch_size)


        # calculate numbers of data with and without unlabled data
        self.exact_number_labeled = len(self.file_names_with_labels)  # complete dataset
        # calculate average usage of unlabeled data and compare with maximum
        self.max_percent_unlabeled_data_in_batch = max_percent_unlabeled_data_in_batch

        self.exact_number_unlabeled = len(self.file_names_without_lables )  # complete  unlabeled dataset

        # # get number of all files
        self.exact_number = len(self.file_names)  # complete dataset

        assert self.exact_number > 0, "Please ensure that you specified the correct loading directory, no data was found at %s" % directory

        if self.exact_number < self.batch_size:
            print("Warning: Use bs of %d, because not enough samples were provided" % self.exact_number)
            self.batch_size = self.exact_number


        average_usage_unlabeled = self.exact_number_unlabeled / (
                    self.exact_number_labeled + self.exact_number_unlabeled)  # percentage of labeled data vs. all data
        if average_usage_unlabeled <= max_percent_unlabeled_data_in_batch:
            # everything fine
            self.use_fixed_ratio_ul_data = False



            self.unlabeled_bs = int (np.round(average_usage_unlabeled * self.batch_size))
            self.labeled_bs = self.batch_size - self.unlabeled_bs

        else:
            # need to load data partially
            self.use_fixed_ratio_ul_data = True
            # mp = 0 -> lbs = 1.0 bs
            # mp = 0.5 -> lbs = 0.5bs
            # mp = 0.8 -> lbs = 0.2bs
            assert max_percent_unlabeled_data_in_batch <= 1 and max_percent_unlabeled_data_in_batch >= 0
            self.labeled_bs = (int)(np.round(self.batch_size * (1 - max_percent_unlabeled_data_in_batch)))
            self.unlabeled_bs = self.batch_size - self.labeled_bs



        self.nb_samples_only_labeled = util.get_nb(self.exact_number_labeled, self.batch_size, silent=True)
        nb_samples_labeled = util.get_nb(self.exact_number_labeled, self.labeled_bs, silent=True)
        nb_samples_unlabeled = util.get_nb(self.exact_number_unlabeled, self.unlabeled_bs, silent=True)

        # ensure same number of epochs
        ep_labeled = nb_samples_labeled // self.labeled_bs
        ep_unlabeled = nb_samples_unlabeled // self.unlabeled_bs if self.unlabeled_bs > 0 else 0

        ep = min(ep_labeled, ep_unlabeled) if ep_unlabeled > 0 else ep_labeled # could be different due to max percent unlabeled data -> equlize nb_samples, unlabeled data can be zero
        nb_samples_labeled = ep * self.labeled_bs
        nb_samples_unlabeled = ep * self.unlabeled_bs

        self.nb_samples = nb_samples_labeled + nb_samples_unlabeled

        print("avg.usage unlabeled %0.02f, nb samples %d [not exact] %d [only-labeled] %d [labled] %d [unlabeled], ep: l - %d ul - %d, bs: l - %d ul - %d" % (average_usage_unlabeled,self.nb_samples, self.nb_samples_only_labeled, nb_samples_labeled, nb_samples_unlabeled, ep_labeled, ep_unlabeled, self.labeled_bs, self.unlabeled_bs))


        # labels for all files
        raw_classes = [file_name.split("/")[-2] for file_name in self.file_names_with_labels] # get class name, expected structure .../class/*.png
        self.classes = sorted(list(set(raw_classes))) # get unique sorted identifiers for classes
        # assert self.classes == class_labels , "%s vs. %s" % (self.classes, class_labels) # must not be true if sub set does not contain all classes
        self.class_to_label_map = dict(zip(self.classes, range(len(self.classes)))) # dictionary/map -> key: class name, value: label number
        self.labels = np.array([ self.class_to_label_map[raw_class] for raw_class in raw_classes]) # get label for every entry




        print("GENERATOR: found %d (sup:%d/unsup:%d) images in %s, use %d per epoch due to batchsize of %d, uses %s alternating" % (
        self.exact_number, len(self.file_names_with_labels), len(self.file_names_without_lables), directory,
        self.nb_samples, self.batch_size, "no" if alternate_count < 0 else ""))
        print("class to label mapping " , self.class_to_label_map)

        # print(self.labels)

        # calculate internal batch size due to sample repetition
        assert self.sample_repetition > 0
        assert self.batch_size % self.sample_repetition == 0
        # assert self.labeled_bs % self.sample_repetition == 0 # not needed any more see below
        # assert self.unlabeled_bs % self.sample_repetition == 0
        self.internal_batch_size = self.batch_size // self.sample_repetition
        self.internal_lbs = (int) (self.labeled_bs // self.sample_repetition)
        self.internal_ulbs = self.internal_batch_size - self.internal_lbs # (int) (self.unlabeled_bs // self.sample_repetition)
        print("Used internal bs: all %d unlabeled %d labeled %d" % (self.internal_batch_size, self.internal_lbs, self.internal_ulbs))

        # internal data generators (augmentator
        self.no_augment_generator = ImageDataGenerator(**no_augments)

        self.imgaug_augments = [] # list of imgaug augmentation for all heads

        # create functions based on names
        for name_aug in imgaug_augments:
            f = function_for_augment_name(name_aug)
            self.imgaug_augments.append(f(target_size))


        # calculate range of indices for partial data and supervision
        self.unlabled_indices = (self.exact_number_labeled + np.arange(self.exact_number_unlabeled)).astype(int)
        self.labled_indices = np.arange(self.exact_number_labeled)

        self.random.shuffle(self.unlabled_indices)
        self.random.shuffle(self.labled_indices)

        self.unlabled_indices = self.unlabled_indices[:self.exact_number_unlabeled]
        self.labled_indices = self.labled_indices[:self.exact_number_labeled]


        # sort them for better usage
        self.labled_indices = np.sort(self.labled_indices)
        self.unlabled_indices = np.sort(self.unlabled_indices)


        print("indices-unlabeled " + np.array2string(self.unlabled_indices))
        print("indices-labeled " + np.array2string(self.labled_indices))

        self.supervised_labels = self.labels

        self.shuffle_data() # shuffle data


        # create item for logging, pair of generated images
        self.logged_generated_images = None
        self.logged_output = None
        self.epoch_counter = 0

        # alternating magic bit for alternating head training

        self.alternate_count = alternate_count # count the epochs, -1 means no alternating
        self.alternate_overcluster_max =alternate_overcluster_max # number of epochs for overcluster head training
        self.alternate_gt_max = alternate_gt_max # number of epochs for gt head training
        self.alternate_max = self.alternate_overcluster_max + self.alternate_gt_max # maximum range for alternating counter





    def __len__(self):
        'Denotes the number of batches per epoch'

        if self.alternate_count >= 0 and self.alternate_count >= self.alternate_overcluster_max:
            return self.nb_samples_only_labeled // self.internal_batch_size
        else:
            return self.nb_samples // self.internal_batch_size

    def shuffle_data(self):
        if self.shuffle:
            # self.random.shuffle(self.indices_1)
            self.random.shuffle(self.unlabled_indices)
            # self.random.shuffle(self.indices_2)
            self.random.shuffle(self.labled_indices)




    def on_epoch_end(self):
        'Updates indexes after each epoch'

        self.shuffle_data()

        if self.alternate_count >= 0:
            self.alternate_count = (self.alternate_count + 1) % self.alternate_max

            print("epoch end, gen update alternate count to ",self.alternate_count )


    def load_images(self, batch_file_names):
        """
        load a given list of images
        :param batch_file_names:
        :return: rgb image of loaded image, and similarities 0 means same , 1 means different
        """
        loaded_image_batch = np.zeros((self.batch_size, self.loading_size, self.loading_size, 3))
        for i, batch_file_name in enumerate(batch_file_names):
            if batch_file_name != "INVALID": # ignore INVALID name
                img = image.load_img(batch_file_name, target_size=(self.loading_size, self.loading_size))
                loaded_image_batch[i] = img

        return loaded_image_batch


    def __getitem__(self, index):
        'Generate one batch of data'


        # load images for first image
        if self.alternate_count >= 0 and self.alternate_count >= self.alternate_overcluster_max:
            # dont use unlabeled indices on normal head
            # print("gen no unlabeled data",self.alternate_count )
            interal_batch_indices_1 = self.labled_indices[index * self.internal_batch_size:(index + 1) * self.internal_batch_size]  # without sample repetition (maybe later duplicating after loading is more efficient)
        else:
            # print("gen include unlabeled data",self.alternate_count )
            interal_batch_indices_1 =  np.concatenate(
                (self.labled_indices[index * self.internal_lbs:(index + 1) * self.internal_lbs],
                self.unlabled_indices[index * self.internal_ulbs:(index + 1) * self.internal_ulbs])
            )
        batch_indices_1 = np.repeat(interal_batch_indices_1, self.sample_repetition) # with sample repetition
        batch_file_names_1 = self.file_names[batch_indices_1]
        loaded_image_batch1 = self.load_images(batch_file_names_1)

        # batch 2
        batch_indices_2 = np.zeros((self.batch_size), dtype=int)
        for i, batch_index in enumerate(batch_indices_1):
            if batch_indices_1[i] in self.labled_indices:  # check batch index in supervised index
                target_label = self.labels[batch_indices_1[i]]

                # get a random indice with the same label
                same_label_ind = np.where(self.supervised_labels == target_label)[0]  # structure: ([indices])
                random_index = self.random.choice(same_label_ind)
                batch_indices_2[i] = self.labled_indices[random_index]
            else:
                batch_indices_2[i] = batch_indices_1[i]

        loaded_image_batch2 = self.load_images(self.file_names[batch_indices_2])

        # load a third image batch with different images

        batch_indices_3 = np.zeros((self.batch_size), dtype=int)
        for i, batch_index in enumerate(batch_indices_1):
            if batch_indices_1[i] in self.labled_indices:  # check batch index in supervised index
                target_label = self.labels[batch_indices_1[i]]

                # get a random indice with different label
                diff_label_ind = np.where(self.labels != target_label)[0]  # structure: ([indices])
                if len(diff_label_ind) > 0 :
                    random_index = self.random.choice(diff_label_ind)
                    batch_indices_3[i] = random_index
                else:
                    batch_indices_3[i] = self.random.choice(
                        np.concatenate((self.labled_indices, self.unlabled_indices)))  # get a random image
            else:
                batch_indices_3[i] = self.random.choice(np.concatenate((self.labled_indices, self.unlabled_indices))) # get a random image
                # batch_indices_3[i] = -1 # get a black image

        # batch3_names = np.concatenate((self.file_names , np.array(["INVALID"]))) # last label is invalid -> leads to black image
        batch3_names = self.file_names

        loaded_image_batch3 = self.load_images(batch3_names[batch_indices_3])



        # augmentation with imaug
        img_batch_1 = np.array(self.imgaug_augments[0](images=loaded_image_batch1.astype(np.uint8)))
        img_batch_2 = np.array(self.imgaug_augments[1](images=loaded_image_batch2.astype(np.uint8)))

        img_batch_1 = self.no_augment_generator.flow(img_batch_1, batch_size=self.batch_size, shuffle=False)[0]
        img_batch_2 = self.no_augment_generator.flow(img_batch_2, batch_size=self.batch_size, shuffle=False)[0]

        # augmentation with imaug
        loaded_image_batch3 = np.array(self.imgaug_augments[2](images=loaded_image_batch3.astype(np.uint8)))
        img_batch_3 = self.no_augment_generator.flow(loaded_image_batch3, batch_size=self.batch_size, shuffle=False)[0]



        # calculate output distribution
        # output is gt label distribution or zeros
        y = []

        for i in range(self.subheads_number):
            if i < self.overcluster_subheads_number:
                # overcluster head -> dont use supervision
                y_head = np.zeros((self.batch_size, self.overcluster_num_classes))

                for j in range(len(batch_indices_1)):
                    if batch_indices_1[j] in self.labled_indices:  # check batch index in supervised index
                        y_head[j,:] = 1 # just indicate its supervised


            else:
                # gt cluster head
                y_head = np.zeros((self.batch_size, self.num_classes))

                # file in supervised images in this batch
                for j in range(len(batch_indices_1)):
                    if batch_indices_1[j] in self.labled_indices:  # check batch index in supervised index
                        # first images are supervised
                        y_head[j,self.labels[batch_indices_1[j]]] = 1


            y.append(y_head)



        if self.alternate_count >= 0:

            images = [img_batch_1, img_batch_2,img_batch_3]

            if self.alternate_count < self.alternate_overcluster_max:
                return images,y[:self.overcluster_subheads_number]
            else:
                return images, y[self.overcluster_subheads_number:]


        return [img_batch_1, img_batch_2, img_batch_3], y

    def set_alternate_count(self, count):
        self.alternate_count = count
