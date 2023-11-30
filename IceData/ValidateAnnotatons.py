# The purpose of this script is to validate that the VIA image
# annotations are correct, this becomes a problem when the user
# builds annotatons in multiple steps
# 
# The idea is that we can check the annotations (or combine them) without
# having to start a training set.

import os
import json

#paths
def _initpaths ():
    ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
    _annotations_dir=ROOT_DIR+'\\IceData\\NRC_data_multi_stage_big\\train\\'
    # _annotations_dir=ROOT_DIR+'\\IceData\\NRC_data_multi_stage_small\\train\\'
    RegionAttributeName='Object'
    return [ROOT_DIR,_annotations_dir,RegionAttributeName]

def VIA_annotation_val(ROOT_DIR,_annotations_dir,RegionAttributeName): 
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
    try:
        # annotations = json.load(open(os.path.join(_annotations_dir,'via_region_data.json')))   
        # annotations = list(annotations.values())  # don't need the dict keys
        annotations = json.load(open(os.path.join(_annotations_dir,'via_region_data_prjct.json'))) 
        a2=annotations['_via_img_metadata']
        annotations=list(a2.values())
        
    except: 
        print("\n \n \n Ensure json annotatons file is labelled: via_region_data.json (not found) \n \n \n")
        exit()
    # The VIA tool saves images in the JSON even if they don't have any
    # annotations. Skip unannotated images.
    annotations = [a for a in annotations if a['regions']]

    # Add images
    Missing_RegionAttribute={}
    MissingFiles=[]
    for a in annotations:
        filename=a['filename']
        # Get the x, y coordinaets of points of the polygons that make up
        # the outline of each object instance. These are stores in the
        # shape_attributes (see json format above)
        # The if condition is needed to support VIA versions 1.x and 2.x.
        
        # class_labels = [r['region_attributes'][RegionAttributeName] for r in a['regions']]
        for i in a['regions']:
            try:   
                val=i['region_attributes'][RegionAttributeName]
            
            except:
                Missing_RegionAttribute[filename+'_xpts']=i['shape_attributes']['all_points_x']
            

        image_path = os.path.join(_annotations_dir, filename)
        
        if not os.path.exists(image_path):
            MissingFiles.append(filename)
        
    if len(Missing_RegionAttribute)==0 and len(MissingFiles)==0:
        print("\nAll good\n")
    else:
        if len(Missing_RegionAttribute)>0: 
            print("\nmissing region attributes: ", Missing_RegionAttribute)
        if len(MissingFiles)>0: 
            print("\nmissing Files: ", MissingFiles)
            
if __name__ == '__main__':
    ROOT_DIR,_annotations_dir,RegionAttributeName = _initpaths()
    VIA_annotation_val(ROOT_DIR,_annotations_dir,RegionAttributeName)
    