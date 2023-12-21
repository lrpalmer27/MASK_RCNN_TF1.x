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
import matplotlib
import matplotlib.pyplot as plt
import skimage.draw
from skimage.io import imsave

# load the class label names from disk, one label per line
# CLASS_NAMES = open("coco_labels.txt").read().strip().split("\n")

ROOT_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASS_NAMES = ['BG', 'Ice','Ship']
TestDir=os.path.join(ROOT_DIR,'IceData','test_imgs')
TrainedWeights=os.path.join(ROOT_DIR,'logs','supercompDec08','mask_rcnn_dec08_moreimgs_0100.h5')
# TrainedWeights=os.path.join(ROOT_DIR,'logs',"dec12_nrc_25mice","mask_rcnn_dec12_nrc_25mice_0020.h5")
# TrainedWeights=os.path.join(ROOT_DIR,'logs',"dec12_nrc_12_5mice","mask_rcnn_dec12_nrc_12_5mice_0009.h5")
TrainedWeights=os.path.join(ROOT_DIR,'logs','wholemodel_w_augmentation_dec15',"mask_rcnn_wholemodel_w_augmentation_dec15_0391.h5")


def visualize (image,r,save=False,path=None):
    if not save:
        mrcnn.visualize.display_instances(image=image, 
                                        boxes=r['rois'], 
                                        masks=r['masks'], 
                                        class_ids=r['class_ids'], 
                                        class_names=CLASS_NAMES, 
                                        scores=r['scores'])
    else:
        matplotlib.use("Agg")
        height=image.shape[0]
        width=image.shape[1]
        shape=(int(width/120),int(height/120))
        fig,ax=mrcnn.visualize.display_instances(image=image, 
                                        boxes=r['rois'], 
                                        masks=r['masks'], 
                                        class_ids=r['class_ids'], 
                                        class_names=CLASS_NAMES, 
                                        scores=r['scores'],
                                        auto_show=False,
                                        figsize=shape)
        # fig.imshow() 
        # fig.show()
        
        fig.savefig(path)

class SimpleConfig(mrcnn.config.Config):
    # Give the configuration a recognizable name
    NAME = "Ice_ship_interference"
    
    # set the number of GPUs to use along with the number of images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    DETECTION_MAX_INSTANCES = 800
    
    # IMAGE_MAX_DIM=1024
    # IMAGE_MIN_DIM=1024
    
    DETECTION_NMS_THRESHOLD = 0.1
    
    DETECTION_MIN_CONFIDENCE = 0.3 #lowering this lets us see more low prob detections -- usually OK

	# Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
    NUM_CLASSES = len(CLASS_NAMES)

automatic=True
if automatic:
    # makepics={'dec12_nrc_12_5mice':'mask_rcnn_dec12_nrc_12_5mice_0009.h5',
    #         'dec12_nrc_25mice':'mask_rcnn_dec12_nrc_25mice_0020.h5',
    #         'dec12_nrc_50mice':'mask_rcnn_dec12_nrc_50mice_0085.h5',
    #         "dec12_nrc_100mice":'mask_rcnn_dec12_nrc_100mice_0026.h5'}

    makepics={'megamodel_prt1':'mask_rcnn_augmentedmegamodel_0140.h5'}

    model = mrcnn.model.MaskRCNN(mode="inference", 
                                config=SimpleConfig(),
                                model_dir=os.getcwd())
    
    testDir=os.path.join(ROOT_DIR,'IceData','test_imgs')
    testImgs=os.listdir(testDir)
    saveDir=os.path.join(ROOT_DIR,'currentStatusImgs')

    for dire in list(makepics.keys()): 
        TrainedWeights=os.path.join(ROOT_DIR,'logs',f'{dire}',makepics[dire])
        model.load_weights(filepath=TrainedWeights,by_name=True)
        
        for img in testImgs:
            imgpth=os.path.join(testDir,img)
            image = cv2.imread(imgpth) #picks a random image in the kangaroo test image dir.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            r = model.detect([image], verbose=0)
            r = r[0]
            
            visualize(image,r,save=True,path=os.path.join(saveDir,f'{dire}_{img}'))

else:
    # Initialize the Mask R-CNN model for inference and then load the weights.
    # This step builds the Keras model architecture.
    model = mrcnn.model.MaskRCNN(mode="inference", 
                                config=SimpleConfig(),
                                model_dir=os.getcwd())

    # Load the weights into the model.
    model.load_weights(filepath=TrainedWeights, 
                    by_name=True)

    # load the input image, convert it from BGR to RGB channel
    imgpath=r"C:\Users\logan\Desktop\MEng\Mask_RCNN\IceData\test_imgs\12p5m_9ths_0p5kts_0p6m_0deg_001_c_overhead_frame346.png"
    image = cv2.imread(imgpath) #picks a random image in the kangaroo test image dir.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform a forward pass of the network to obtain the results
    r = model.detect([image], verbose=0)

    # Get the results for the first image.
    r = r[0]

    # Visualize the detected objects.
    print('\n\nChosen random file to display with mask predictions: ',imgpath)

    visualize (image,r) #save fcn doesnt work yet