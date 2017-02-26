# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 23:20:23 2017
@author: Johnny

Kaggle Data Science Bowl 2017
Preprocessing code for DICOM prior to applying machine learning
Some functions taken/modified from kaggle kernel from Guido Zuidhof 

"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom #load dicom images, and import class to work with dicom metadata
import os
import scipy.ndimage
import matplotlib.pyplot as plt
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


#load the raw/dicom data
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices


#extract 3d np array from dicom loaded file; output values in Hounsfield Units (HU)
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


#isotropic resampling to normalize spacing
def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing

#optional 3d plotting/ often crashes on my slow laptop
def plot_3d(image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    
    verts, faces = measure.marching_cubes(p, threshold) #Marching cubes is an algorithm to extract a 2D surface mesh from a 3D volume

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
    
    
    
#segmentation to isolate lung tissue
"""
# * Threshold the image (-320 HU is a good threshold, but it doesn't matter much for this approach)
# * Do connected components, determine label of air around person, fill this with 1s in the binary image
# * Optionally: For every axial slice in the scan, determine the largest solid connected component (the body+air around the person), and set others to 0. This fills the structures in the lungs in the mask.
# * Keep only the largest air pocket (the human body has other pockets of air here and there)."""

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None



def segment_lung_mask(image, fill_lung_structures=True):
    
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]
    
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
 
    return binary_image



#normalize the data: Our values currently range from -1024 to around 2000; >400 is not interesting to us (bones with different radiodensity)
def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image


# Zero centering
""" As a final preprocessing step, zero center your data so that your mean value is 0 (subtract the mean pixel value from all pixels).
# To determine this mean you simply average all images in the whole dataset. Do not zero center with the mean per image!  mean ~ 0.25 in the LUNA16 competition. 
"""
def zero_center(image):
    image = image - PIXEL_MEAN
    return image











if __name__ == "__main__":
    
    # Some constants 
    plot_data = 0
    psave = 1
    BASE_FOLDER = r'D:\kaggle\lungcancer\preprocessed_sample_images/'
    INPUT_FOLDER = r'D:\kaggle\lungcancer\sample_images/' #sample images only
    #INPUT_FOLDER = r'H:\kaggle\lung_cancer/' #stage1 images

    
    #parameters for normalization
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    PIXEL_MEAN = 0.25
    
    #load patient data
    patients = os.listdir(INPUT_FOLDER)
    patients.sort()
    
    #process data
    counter = 1
    for patient in patients:
        
        print("loading patient {} of {}: {}".format(counter, len(patients), patient))
        data = load_scan(INPUT_FOLDER + patients[0])
        stack = get_pixels_hu(data)
        if plot_data == 1:
            plt.hist(stack.flatten(), bins=80, color='c')
            plt.xlabel("Hounsfield Units (HU)")
            plt.ylabel("Frequency")
            plt.show()
      
        # when resampling, save the new spacing! Due to rounding this may be slightly off from the desired spacing  
        pix_resampled, spacing = resample(stack, data, [1,1,1]) #resample our patient's pixels to an isomorphic resolution of 1 by 1 by 1 mm
        print("Shape before and after resampling {} to {}".format(stack.shape, pix_resampled.shape))
        
        #segment just the lungs
        #segmented_lungs = segment_lung_mask(pix_resampled, False)
        segmented_lungs_fill = segment_lung_mask(pix_resampled, True)
        
        for i in range(pix_resampled.shape[0]): #go through z-stack
            mask = segmented_lungs_fill[i,:,:]
            pix_masked = pix_resampled[i].copy() #consider removing the copy to save computational time
            pix_masked[mask == 0] = -1024
                      
        if psave == 1:
            outfile = patient + '.npy'
            outpath = os.path.join(BASE_FOLDER, outfile)
            print("saving preprocessed file: {}".format(outfile))
            np.save(outpath, pix_masked)
                
        counter += 1