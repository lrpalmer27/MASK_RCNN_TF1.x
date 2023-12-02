# """
# ***********Based on the balloon samples**********

# Mask R-CNN
# Train on the toy Ice dataset and implement color splash effect.

import os
import sys
import json
import numpy as np
import skimage.draw

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
restarting = False #Change this to True if you want to grab the saved weights in the .logs dir and keep training.
dataset_path=os.path.join(ROOT_DIR,'IceData','NRC_data_multi_stage_big')
model_path = os.path.join(ROOT_DIR,'mrcnntf114_big_weights.h5') #path where you want model saved

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

##"hiding" these paths down here; shouldnt regularily change, keeping away from top of script.
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5") #path to coco weights
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR,'logs') #path to directory to store logs:


############################################################
#  Configurations
############################################################

class IceConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = 'playtest'

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 + 1 # Background + Ice + Ship

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 75 #100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    
    #since our images are huge
    IMAGE_MAX_DIM=1024

############################################################
#  Dataset
############################################################

class IceDataset(utils.Dataset):

    def load_Ice(self, dataset_dir, subset):
        """Load a subset of the Ice dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("IceShipItem", 1, "Ice")
        self.add_class("IceShipItem", 2, "Ship")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        
        self.load_annotations_VGG(dataset_dir,subset)
        
    def load_annotations_VGG(self,dataset_dir,subset):
        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        
        # annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json"))) ## see next line why thisis commented out
        # annotations = list(annotations.values())  # don't need the dict keys
        
        #this allows me to directly save the
        #VIA project files to the directory and not have to worry about extracting the "annoations", this just makes life easier if 
        # I plan on adding more annotations later
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data_prjct.json"))) 
        a2=annotations['_via_img_metadata']
        annotations=list(a2.values())

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
                class_labels = [r['region_attributes']['Object'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]
                class_labels = [r['region_attributes']['Object'] for r in a['regions']] 

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            
            ## Not very pythonic way to do this, but it works.
            ids= [item.get('name') for item in self.class_info]
            class_ids=[]

            for l in class_labels:
                for id in range(len(ids)):
                    if l==ids[id]:
                        class_ids.append(id)

            self.add_image(
                "IceShipItem",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons, class_ids=np.array(class_ids))

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a Ice dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "IceShipItem": #"Trial" needs to match the source name defined above"
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        # return mask.astype(bool), np.ones([mask.shape[-1]], dtype=np.int32)
        
        # Return mask, and array of class IDs of each instance.
        # info['class_labels'] #doesnt work
        return mask, info['class_ids']

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "Ice":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model,dataset_path):
    """Train the model."""
    # Training dataset.
    dataset_train = IceDataset()
    # dataset_train.load_Ice(args.dataset, "train")
    dataset_train.load_Ice(dataset_path, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = IceDataset()
    dataset_val.load_Ice(dataset_path, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=400,
                layers='heads')

if __name__ == '__main__':
    config = IceConfig()
    config.display()
    model = modellib.MaskRCNN(mode="training", config=config,model_dir=DEFAULT_LOGS_DIR)
    
    if restarting: 
        weights_path = model.find_last() #uncomment if need to re-start after pausing training.
        model.load_weights(weights_path, by_name=True)
    else: 
        weights_path = os.path.join(ROOT_DIR,"mask_rcnn_coco.h5")
        model.load_weights(weights_path, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc","mrcnn_bbox", "mrcnn_mask"])

    train(model,dataset_path)  
    model.keras_model.save_weights(model_path) #saves the final weights in the main working folder.
   