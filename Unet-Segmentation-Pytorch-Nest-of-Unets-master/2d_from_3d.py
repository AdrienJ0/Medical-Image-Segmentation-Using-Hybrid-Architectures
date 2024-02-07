import cv2
import scipy.misc
import numpy
import imageio

import SimpleITK as sitk #reading MR images
import os
import glob


original_synapse_path = 'data/original_synapse_dataset/'
modified_2d_synapse_path = 'data/2d_synapse_dataset/'

if not os.path.exists(modified_2d_synapse_path):
 os.makedirs(modified_2d_synapse_path)

readfolderT = glob.glob(original_synapse_path + 'averaged-training-images/*.nii.gz')
readfolderL = glob.glob(original_synapse_path + 'averaged-training-labels/*.nii.gz')
readfolderTest = glob.glob(original_synapse_path + 'averaged-testing-images/*.nii.gz')


TrainingImagesList = []
TrainingLabelsList = []
TestingImagesList = []

for i in range(len(readfolderT)):
    y_folder = readfolderT[i]
    yread = sitk.ReadImage(y_folder)
    yimage = sitk.GetArrayFromImage(yread)
    x = yimage[:184,:232,112:136]
    x = numpy.rot90(x)
    x = numpy.rot90(x)
    for j in range(x.shape[2]):
        TrainingImagesList.append((x[:184,:224,j]))

for i in range(len(readfolderL)):
    y_folder = readfolderL[i]
    yread = sitk.ReadImage(y_folder)
    yimage = sitk.GetArrayFromImage(yread)
    x = yimage[:184,:232,112:136]
    x = numpy.rot90(x)
    x = numpy.rot90(x)
    for j in range(x.shape[2]):
        TrainingLabelsList.append((x[:184,:224,j]))
        
for i in range(len(readfolderTest)):
    y_folder = readfolderTest[i]
    yread = sitk.ReadImage(y_folder)
    yimage = sitk.GetArrayFromImage(yread)
    x = yimage[:184,:232,112:136]
    x = numpy.rot90(x)
    x = numpy.rot90(x)
    for j in range(x.shape[2]):
        TestingImagesList.append((x[:184,:224,j]))

for i in range(len(TrainingImagesList)):

    xchangeL = TrainingImagesList[i]
    xchangeL = cv2.resize(xchangeL,(128,128))
    if not os.path.exists(modified_2d_synapse_path + '2d_training_images/'):
        os.makedirs(modified_2d_synapse_path + '2d_training_images/')
    imageio.imwrite(modified_2d_synapse_path + '2d_training_images/' + str(i) + '.png', xchangeL)

for i in range(len(TrainingLabelsList)):

    xchangeL = TrainingLabelsList[i]
    xchangeL = cv2.resize(xchangeL,(128,128))
    if not os.path.exists(modified_2d_synapse_path + '2d_training_labels/'):
        os.makedirs(modified_2d_synapse_path + '2d_training_labels/')
    imageio.imwrite(modified_2d_synapse_path + '2d_training_labels/' + str(i) + '.png', xchangeL)
    
for i in range(len(TestingImagesList)):

    xchangeL = TestingImagesList[i]
    xchangeL = cv2.resize(xchangeL,(128,128))
    if not os.path.exists(modified_2d_synapse_path + '2d_testing_images/'):
        os.makedirs(modified_2d_synapse_path + '2d_testing_images/')
    imageio.imwrite(modified_2d_synapse_path + '2d_testing_images/' + str(i) + '.png', xchangeL)