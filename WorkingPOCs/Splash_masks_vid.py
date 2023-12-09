# pulled basics from splash of colour example

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
import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import skimage.draw
from skimage.io import imsave
import math
import io
from PIL import Image

ROOT_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASS_NAMES = ['BG', 'Ice','Ship']
TestDir=os.path.join(ROOT_DIR,'IceData','test_imgs')
TrainedWeights=os.path.join(ROOT_DIR,'logs','super_dec04_lowsteps','mask_rcnn_maindec05_lowsteps_0050.h5')
videoFILEpath=os.path.join(ROOT_DIR,'IceData','SplashVideos','50m_9ths_1p2kts_0p6m_0deg_001_cropped_lowQuality.mp4')

class SimpleConfig(mrcnn.config.Config):
    # Give the configuration a recognizable name
    NAME = "Interference_video"
    IMAGES_PER_GPU = 1 #1 for dev 3 for implementation.
    
	# Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
    NUM_CLASSES = len(CLASS_NAMES)

def color_splash(image, mask):
    # pulled basics from splash of colour example
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash

def DetectAndMask(frame,r,shape,ShowCentroids=True,ShowRegions=True):
    matplotlib.use("Agg") #backend of matplotlib used for non graphical use
    width,height=shape
    shape=(int(width/120),int(height/120))
    lightBlue=[255/255,153/255,255/255]
    colors=[lightBlue for i in range(0,r['rois'].shape[0])]
    fig,ax=mrcnn.visualize.display_instances(image=frame, 
                                        boxes=r['rois'], 
                                        masks=r['masks'], 
                                        class_ids=r['class_ids'], 
                                        class_names=CLASS_NAMES, 
                                        scores=r['scores'],
                                        auto_show=False,
                                        figsize=shape,
                                        colors=colors)
    
    # plt.show() #this shows the masked image
    # print(type(ax))
    # print(type(_))
    ax.axis('off')
    
    if ShowCentroids or ShowRegions: 
        import WorkingPOCs.ExtractData_fromMRCNN_toCSV as ED
        regionstats,_=ED.processDetections(r,CLASS_NAMES=CLASS_NAMES,verbose=False)
        ax=ED.viz_centroids(regionstats['Centroids'],r,plt=ax,CLASS_NAMES=CLASS_NAMES,ShowCentroids=ShowCentroids,ShowRegions=ShowRegions,showimg=False)
    
    ##down here deals with getting the matlab figure into a numpy array so we can keep rolling with
    ## it later
    # matplotlib.use("Agg") #backend of matplotlib used for non graphical use
    fig.canvas.draw()
    rgba_buf = fig.canvas.buffer_rgba()
    (w,h) = fig.canvas.get_width_height()
    rgba_arr = (np.frombuffer(rgba_buf, dtype=np.uint8).reshape((h,w,4))).astype(np.uint8)
    
    rgb_arr=rgba_arr[:,:,:3]
    out=cv2.resize(rgb_arr, (width, height))
    return out


def detect_and_color_splash(model, image_path=None, video_path=None,SplashOfColour=True,OverlayDetections=True):
    # pulled basics from splash of colour example
    assert image_path or video_path
    
    # Image or video?
    if image_path:
        BasePath=os.path.dirname(image_path)
        # Run model detection and generate the color splash effect
        print("Running on {}".format(os.path.basename(image_path)))
        # Read image
        image = skimage.io.imread(image_path)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = os.path.join(BasePath,"splashOfColor_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now()))
        skimage.io.imsave(file_name, splash)
        
    elif video_path:
        BasePath=os.path.dirname(video_path)
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)
        
        if SplashOfColour: ########################################################################################splash of colour
            # Define codec and create video writer
            file_name = os.path.join(BasePath,"splashOfColor_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now()))
            vwriter = cv2.VideoWriter(file_name,
                                    cv2.VideoWriter_fourcc(*'MJPG'),
                                    fps, (width, height))
            count = 0
            success = True
            while success:
                print("frame: ", count)
                # Read next image
                success, image = vcapture.read()
                if success:
                    # OpenCV returns images as BGR, convert to RGB
                    image = image[..., ::-1]
                    # Detect objects
                    r = model.detect([image], verbose=0)[0]
                    # Color splash
                    splash = color_splash(image, r['masks'])
                    # RGB -> BGR to save image to video
                    splash = splash[..., ::-1]
                    # print(type(splash))
                    # Add image to video writer
                    vwriter.write(splash)
                    count += 1
            vwriter.release()
            print("Saved splash of color example to ", file_name)
            
        if OverlayDetections: ########################################################################################OverlayMasks
            # Define codec and create video writer
            file_name = os.path.join(BasePath,"MaskOverlay_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now()))
            vwriter = cv2.VideoWriter(file_name,
                                    cv2.VideoWriter_fourcc(*'MJPG'),
                                    fps, (width, height))
            count = 0
            success = True
            while success:
                print("frame: ", count)
                # Read next image
                success, image = vcapture.read()
                if success:
                    # OpenCV returns images as BGR, convert to RGB
                    image = image[..., ::-1]
                    # Detect objects
                    r = model.detect([image], verbose=0)[0]
                    # OverlayMasksDetections
                    detections=DetectAndMask(frame=image, r=r,shape=(width,height))
                    # print(type(detections))
                    
                    # matplotlib.use("TkAgg") 
                    # f=plt.imshow(detections)
                    # plt.show()
                                       
                    # Add image to video writer
                    vwriter.write(detections)
                    count += 1
            vwriter.release()
            print("Saved overlayed mask detections to ", file_name)
                       
# Initialize the Mask R-CNN model for inference and then load the weights.
model = mrcnn.model.MaskRCNN(mode="inference",config=SimpleConfig(),model_dir=os.getcwd())
model.load_weights(filepath=TrainedWeights, by_name=True)

detect_and_color_splash(model,video_path=videoFILEpath,SplashOfColour=False,OverlayDetections=True)