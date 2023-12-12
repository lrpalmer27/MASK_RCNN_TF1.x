import sys
import os
ROOT_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import skimage.draw
from skimage.io import imsave

# load the class label names from disk, one label per line
# CLASS_NAMES = open("coco_labels.txt").read().strip().split("\n")

ROOT_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASS_NAMES = ['BG', 'Ice','Ship']
TestDir=os.path.join(ROOT_DIR,'IceData','test_imgs')
# TrainedWeights=os.path.join(ROOT_DIR,'logs','super_dec04_lowsteps','mask_rcnn_maindec05_lowsteps_0050.h5')
TrainedWeights=os.path.join(ROOT_DIR,'logs','superC_dec11_smallIce','mask_rcnn_dec11_smallicetrainingonly_0073.h5')


def visualize (image,r,save=False):
    if save==False:
        mrcnn.visualize.display_instances(image=image, 
                                        boxes=r['rois'], 
                                        masks=r['masks'], 
                                        class_ids=r['class_ids'], 
                                        class_names=CLASS_NAMES, 
                                        scores=r['scores'])
    else: 
        #this saves the file in the IceData dir to ref later (cant have things pop up on alliance can clusters)
        masks=r['masks']
        nmask = (np.sum(masks, -1, keepdims=True) >= 1)
        ngray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
        nsplash = np.where(nmask, image,ngray).astype(np.uint8)
        imsave(os.path.join(ROOT_DIR,'test.png'),nsplash)
        pass 

class SimpleConfig(mrcnn.config.Config):
    # Give the configuration a recognizable name
    NAME = "Ice_ship_interference"
    
    # set the number of GPUs to use along with the number of images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

	# Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
    NUM_CLASSES = len(CLASS_NAMES)

# Initialize the Mask R-CNN model for inference and then load the weights.
# This step builds the Keras model architecture.
model = mrcnn.model.MaskRCNN(mode="inference", 
                             config=SimpleConfig(),
                             model_dir=os.getcwd())

# Load the weights into the model.
model.load_weights(filepath=TrainedWeights, 
                   by_name=True)

# load the input image, convert it from BGR to RGB channel
Test_Dir_list=os.listdir(TestDir) #lists the kangaroo test image dir
randomImg=Test_Dir_list[random.randint(0,len(Test_Dir_list)-1)]
randimgpath=os.path.join(TestDir,randomImg)
randimgpath=r"C:\Users\logan\Desktop\MEng\Mask_RCNN\IceData\test_imgs\25m_9ths_0p5kts_1m_0deg_001_c_overhead_frame476.png"
image = cv2.imread(randimgpath) #picks a random image in the kangaroo test image dir.
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform a forward pass of the network to obtain the results
r = model.detect([image], verbose=0)

# Get the results for the first image.
r = r[0]

# Visualize the detected objects.
print('\n\nChosen random file to display with mask predictions: ',randomImg)

visualize (image,r,save=False) #save fcn doesnt work yet