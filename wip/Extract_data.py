import mrcnn
from mrcnn.config import Config
import mrcnn.model
import mrcnn.visualize
import cv2
import os
import random
import re
import pandas as pd

#important paths 
ROOT_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASS_NAMES = ['BG', 'Ice','Ship']
TestDir="\\IceData\\test_imgs\\"
TrainedWeights=ROOT_DIR+"\\mask_rcnn_iceshiptf1config_0050.h5"
DEFAULT_LOGS_DIR = ROOT_DIR+"\\.logs"
_rawforceData=ROOT_DIR+"\\.rawForcedata"
_rawvidData=ROOT_DIR+"\\.rawVideoData"
_processedData=ROOT_DIR+'\\'+'.processedData'

class loadconfig(Config):
    # Give the configuration a recognizable name
    NAME = "Ice_ship_interference"
    
    # set the number of GPUs to use along with the number of images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

	# Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
    NUM_CLASSES = len(CLASS_NAMES)

def init_model(config):
    # Initialize the Mask R-CNN model for inference and then load the weights.
    # This step builds the Keras model architecture.
    model = mrcnn.model.MaskRCNN(mode="inference", 
                                config=config,
                                model_dir=DEFAULT_LOGS_DIR)

    # Load the weights into the model.
    model.load_weights(filepath=TrainedWeights, 
                    by_name=True)
    
    return model

def RandompickFrame(): 
    # load the input image, convert it from BGR to RGB channel
    Test_Dir=os.listdir(ROOT_DIR+TestDir) 
    Imagepth=Test_Dir[random.randint(0,len(Test_Dir)-1)]
    print('\n\nChosen random file to display with mask predictions: ',Imagepth)

    image = cv2.imread(ROOT_DIR+TestDir+Imagepth) #picks a random image in the kangaroo test image dir.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #cv2 loads img bgr but tf needs it to be in rgb mode.
    
    return image


def detect(model,frame): 
    # Perform a forward pass of the network to obtain the results
    r = model.detect([frame], verbose=0)

    # Get the results for the first image.
    r = r[0]
    
    return r

def visualize(frame,r):
    mrcnn.visualize.display_instances(image=frame, 
                                        boxes=r['rois'], 
                                        masks=r['masks'], 
                                        class_ids=r['class_ids'], 
                                        class_names=CLASS_NAMES, 
                                        scores=r['scores'])
    
def saveResults(frame,r):
    #do things here to save data into processed csv format for each frame of each video.
    print("placeholder")
    
def listbothdirs(): 
    v=os.listdir(_rawvidData)
    r=os.listdir(_rawforceData)
    
    #[expression for item in iterable if condition == True]
    
    videoDir=[i for i in v if not os.path.isdir(_rawvidData+'\\'+i)]
    ForceDir=[f for f in r if not os.path.isdir(_rawforceData+'\\'+f)]
    
    return [sorted(videoDir), sorted(ForceDir)]

def NumFrames(v):
    #from https://github.com/lrpalmer27/Video_frame_extractor
    video_path=_rawvidData+"\\"+v
    try:
        vidObj = cv2.VideoCapture(video_path) 
        number_of_frames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
    except: 
        print("Could not open:",v,"\nFull path:",video_path)
    return number_of_frames

def numframes_forcedata(path):
    #provide path to csv and this will provide number in the last row of column 1
    csvpath=_rawforceData+"\\"+path
    try: 
        df=pd.read_csv(csvpath)
        nframes=df.iat[-1,0]
        
    except: 
        print("Could not open:",path,"\nFull path:",csvpath)
       
    return (int(float(nframes)) + 1)

def GetCSVData(pth):
    #similar to fcn above but return all data (raw)
    csvpath=_rawforceData+"\\"+pth
    try: 
        df=pd.read_csv(csvpath)
        OriginalData=df
        
    except: 
        print("Could not open:",pth,"\nFull path:",csvpath)
      
    return df
    
def validateSize(videoDir, ForceDir): 
    #this fcn validates that the same number of frames are in both the 
    # Step 1: Check if the same files are present in the two dirs
    # Step 2: Check if the are recorded as the same number of frames
    
    #the regex format is the regex of NRC OCRE file format naming conventions
    regex="\d{2,3}p?\d?m_\dths_\d+p\d+kts_\d?p?\dm_\ddeg_\d{3}"
    
    #step1 - do we have the same files in the same dirs?
    forceDir_reMatch = [re.match(regex, f).group() for f in ForceDir if re.match(regex, f)]
    viddir_reMatch = [re.match(regex, v).group() for v in videoDir if re.match(regex, v)]
    
    #make sure we did not miss a file if we didnt need to - ideally this is checked manually for typos infilenames after seeing this output.
    if len(forceDir_reMatch) != len(ForceDir): 
        print("Regex dropped a file from the force input files")
    if len(viddir_reMatch) != len(videoDir): 
        print("Regex dropped a file from the video input files")
    
    for fo in forceDir_reMatch: 
        for vi in viddir_reMatch: 
            if fo==vi: 
                forceDir_reMatch.remove(fo)
                
    if len(forceDir_reMatch) == 0: 
        "All expected videos are present."
    
    #step 2 -- ensure the number of frames matches the expected frames from the dataset csv file.
    different=[]
    for file in ForceDir: 
        name=(re.match(regex,file)).group()
        for vi in videoDir:
            if name in vi: 
                videofile=vi
                videoDir.remove(vi)
                break                 
                
        if NumFrames(videofile) != numframes_forcedata(file): 
            # print("\n\nNumber of frames do not match!!",name)
            different.append(name)
    
    # print("\n\nDifferences in number of frames actual vs recorded!",different)
    
    return different

def processDetections(r): 
    # this function will process the results from the detected model (r) then will pass the new out to then be combined
    # and saved into the updated csv format to pass along to the training stage.
    # this is done on a per frame basis
    None
    
def save_newCSV(OriginalData, processedData,frameNum,filename):
    # this is done on a per frame basis
    filepath=processedData+'\\'+filename+'.csv'
    if not os.path.exists(filepath): #if it doesnt exist yet then put in the basics
        originalheadings=['', 'Carriage_Speed_DP', 'Global_FX', 'Global_FY', 'Global_FT', 'Global_MZ', 'Floe_Size', 'Ice_Conc', 'Drift_Speed', 'Ice_Thick', 'Drift_Angle', 'Trial']
        newheadings=['ConcBw_30_azimuth','ConcBw_90_azimuth','ConcBw_180_azimuth']
        with open(filepath,'w') as file:
            writer=csv.writer(file)
            writer.writerow(originalheadings+newheadings)
        lastrow=0+1 #alternate to opening the file we just created, we can just set lastrow to 0.
        # add 1 to make sure we dont overwrite the last row - ie. last row is the row we CAN write on
    else: 
        lastrow=len(list(csv.reader(open(filename))))+1 #add 1 to make sure we dont overwrite the last row - ie. last row
        # is the row we CAN write on
    
    # Now we can add the original data, and new data; frame by frame (this fcn gets called each frame)
    with open(filepath,'w') as file:
            writer=csv.writer(file)
            writer.writerow(OriginalData+processedData)
    
    None

if __name__ == "__main__": 
    config=loadconfig()
    config.display()
    [videoDir, ForceDir]=listbothdirs()
    diff=validateSize(videoDir, ForceDir)
    if len(diff) != 0: 
        print("Look into these differences",diff)
    ## up to here is working; in the test dataset - the videos and force containing csv files all have the same number of frames recorded.
    
    ## TODO (WIP): Now add iterative method to grab each datafile (CSV), extract the data we want to train a model on then rebuild a csv file for that video instance.
    mdl=init_model(config)
    for instance in ForceDir:
        regex="\d{2,3}p?\d?m_\dths_\d+p\d+kts_\d?p?\dm_\ddeg_\d{3}"
        filename=re.match(regex, instance).group() #this extracts the important stuff not the garbage extras that are incl.
        OriginalData=GetCSVData(instance) #returns a pandas dataframe.
        for frame in numframes_forcedata(instance):
            r=detect(mdl,frame)
            proc=processDetections(r) #TODO build this
            save_newCSV(OriginalData,proc,frame,filename) #TODO build this
            
 
    # mdl=init_model(config)
    # frm=RandompickFrame(instance)
    # r=detect(mdl,frm)
    # visualize(frm,r) #vis
    
    
