#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 23:20:23 2017
@author: Johnny

Kaggle Data Science Bowl 2017
Preprocessing code for DICOM prior to applying machine learning
Some functions taken/modified from kaggle kernel from Guido Zuidhof

"""

from __future__ import print_function, division
import numpy as np  # linear algebra
#  import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom as dicom  # load dicom images, class to work with dicom metadata
import os
import scipy.ndimage
import matplotlib.pyplot as plt
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from keras.models import Sequential
from keras.layers import Dense, ConvLSTM2D, BatchNormalization, \
                         Convolution2D, TimeDistributed, Flatten
# use Theano backend

'''********************* Dataset I/O ****************************'''


# load the raw/dicom data
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] -
                                 slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation -
                                 slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def load_ground_truth(path):
    ground_truth = open(path)
    ground_truth_dict = {}
    for line in ground_truth:
        line = [word.strip() for word in line.strip().split(',')]
        ground_truth_dict.update([line])
    return ground_truth_dict

'''********************* Preprocessing ****************************'''


# extract 3d np array from dicom loaded file;
# output values in Hounsfield Units (HU)
def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)

# segmentation to isolate lung tissue
"""
# * Threshold the image (-320 HU is a good threshold, but it doesn't matter much for this approach)
# * Do connected components, determine label of air around person, fill this with 1s in the binary image
# * Optionally: For every axial slice in the scan, determine the largest solid connected component (the
# * body+air around the person), and set others to 0. This fills the structures in the lungs in the mask.
# * Keep only the largest air pocket (the human body has other pockets of air here and there)."""


def largest_label_volume(labels, bg=0):
    vals, counts = np.unique(labels, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if counts.any():  # empty array is false
        return vals[np.argmax(counts)]
    else:
        return None


def segment_lung_mask(image, fill_lung_structures=True):
    LUNG_HU_VAL = -320
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image < LUNG_HU_VAL, dtype=np.int8)
    labels = measure.label(binary_image, connectivity=2)

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = labels[0, 0, 0]

    # Fill the air around the person
    binary_image[labels == background_label] = 0

    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        l_max = largest_label_volume(labels, bg=0)
        if l_max is not None:  # This slice contains some lung
            mask = np.zeros_like(binary_image)
            mask[labels == l_max] = 1

    # Remove other air pockets insid of body
    mask = morphology.binary_closing(mask)

    return mask


# normalize the data: Our values currently range from -1024 to around 2000; >400 is not interesting to us (bones with different radiodensity)
def normalize(image, MIN_BOUND, MAX_BOUND):
    im = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    im[im > 1] = 1.
    im[im < 0] = 0.
    return im


# Zero centering
""" As a final preprocessing step, zero center your data so that your mean value is 0 (subtract the mean pixel value from all pixels).
# To determine this mean you simply average all images in the whole dataset. Do not zero center with the mean per image!  mean ~ 0.25 in the LUNA16 competition.
"""


def zero_center(image):
    image = image - PIXEL_MEAN
    return image


def pad_with_zeros(image_stack, final_dims):
    new_stack = np.zeros(final_dims)
    z, x, y = current_shape = image_stack.shape
    dz, dx, dy = np.subtract(new_stack.shape, current_shape)//2
    new_stack[dz:dz+z, dx:dx+x, dy:dy+y] = image_stack
    return new_stack


# isotropic resampling to normalize spacing
def resample(image, scan, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing,
                       dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image,
                                             real_resize_factor,
                                             mode='nearest')

    return image, new_spacing

'''********************* Plotting ****************************'''


# optional 3d plotting/ often crashes on my slow laptop
def plot_3d(image, threshold=-300):

    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)

    verts, faces = measure.marching_cubes(p, threshold)  # Marching cubes is an algorithm to extract a 2D surface mesh from a 3D volume

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.7)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


'''********************* Model ****************************'''


def build_model():
    seq = Sequential()
    seq.add(ConvLSTM2D(nb_filter=40, nb_row=3, nb_col=3,
                       input_shape=(512, 512, 512, 1),
                       border_mode='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(nb_filter=40, nb_row=3, nb_col=3,
                       border_mode='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(nb_filter=40, nb_row=3, nb_col=3,
                       border_mode='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(nb_filter=40, nb_row=3, nb_col=3,
                       border_mode='same', return_sequences=False))
    seq.add(BatchNormalization())
    seq.add(Flatten())

    seq.add(Dense(100, activation='tanh'))
    seq.add(Dense(100, activation='tanh'))
    seq.add(Dense(1, activation='sigmoid'))

    seq.compile(loss='binary_crossentropy', optimizer='nadam')

    return seq


if __name__ == "__main__":

    # Some constants
    PLOT_DATA = 1
    PSAVE = 0
    BASE_FOLDER = '/Users/ABMBP/Documents/Kaggle/Lung_cancer'
    GROUND_TRUTH_PATH = BASE_FOLDER + '/stage1_labels.csv'
    INPUT_FOLDER = BASE_FOLDER + '/stage1'
    # INPUT_FOLDER = r'H:\kaggle\lung_cancer/' #stage1 images

    # parameters for normalization
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    PIXEL_MEAN = 0.25

    # load patient data
    patients = os.listdir(INPUT_FOLDER)[1:]  # get rid of .DS_store file
    patients.sort()

    # process data
    counter = 1
    max_dims = [0, 0, 0]

    ground_truth_dict = load_ground_truth(GROUND_TRUTH_PATH)

    N = 1
    for patient in patients[:N]:
        print("loading patient {} of {}: {}".format(counter,
                                                    len(patients),
                                                    patient))
        data = load_scan(INPUT_FOLDER + '/' + patient)
        ground_truth = float(ground_truth_dict[patient])
        stack = get_pixels_hu(data)

#        if plot_data == 1:
#            plt.hist(stack.flatten(), bins=80, color='c')
#            plt.xlabel("Hounsfield Units (HU)")
#            plt.ylabel("Frequency")
#            plt.show()

        # when resampling, save the new spacing! Due to rounding this may be slightly off from the desired spacing
        pix_resampled, spacing = resample(stack, data, [1, 1, 1])  # resample our patient's pixels to an isomorphic resolution of 1 by 1 by 1 mm

        max_dims = np.maximum(pix_resampled.shape, max_dims)  # save max dims

        print("Shape before and after resampling {} to {}".format(stack.shape, pix_resampled.shape))

        # segment just the lungs
        # segmented_lungs = segment_mlung_mask(pix_resampled, False)
        lung_mask = segment_lung_mask(pix_resampled, True)
        pix_normalized = normalize(pix_resampled, MIN_BOUND, MAX_BOUND)
        pix_masked = pix_normalized * lung_mask
        pix_padded = pad_with_zeros(pix_masked, [512, 512, 512])

        # plot an example frame
        if PLOT_DATA:
            i = 75  # frame #

            f, axarr = plt.subplots(2, 2)
            # axarr[0,0].imshow(pix_normalized[i,:,:], cmap=plt.cm.gray) #original image
            axarr[0, 1].imshow(pix_resampled[i, :, :], cmap=plt.cm.gray)  # resampled image
            axarr[1, 0].imshow(lung_mask[i, :, :], cmap=plt.cm.gray)  # mask of resampled image
            axarr[1, 1].imshow(pix_masked[i, :, :], cmap=plt.cm.gray)  # resampled image with mask

            # axarr[0, 0].set_title('pix_normalized', fontsize = 10)
            axarr[0, 1].set_title('resampled', fontsize=10)
            axarr[1, 0].set_title('mask', fontsize=10)
            axarr[1, 1].set_title('mask on resampled', fontsize=10)

            for j in range(2):
                for k in range(2):
                    axarr[j, k].set_xticks([])

        if PSAVE:
            outfile = patient + '.npy'
            outpath = os.path.join(BASE_FOLDER, outfile)
            print("saving preprocessed file: {}".format(outpath))
            np.save(outpath, pix_masked)

        BUILD_MODEL = 1
        if BUILD_MODEL:
            print("Compiling model...\n")
            model = build_model()
            print("Model summary:\n\n", model.summary())

        TRAIN_MODEL = 1
        if TRAIN_MODEL:
            # Model options
            SAVE_MODEL = 0
            BATCH_SIZE = 1
            NB_EPOCHS = 1

            pix_padded = pix_padded[np.newaxis, :, :, :, np.newaxis]
            print("Fitting on image: {}".format(patient))
            model_history = model.fit(pix_padded,
                                      np.array([ground_truth], dtype=np.float16),
                                      batch_size=BATCH_SIZE,
                                      nb_epoch=NB_EPOCHS,
                                      verbose=1)

        counter += 1
