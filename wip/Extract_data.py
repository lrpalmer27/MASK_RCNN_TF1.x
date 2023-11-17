import mrcnn
from mrcnn.config import Config
import mrcnn.model
import mrcnn.visualize
import cv2
import os
import random
import re
import pandas as pd
import numpy as np
import sys
import math
import bisect
import time
import statistics

#important paths 
ROOT_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASS_NAMES = ['BG', 'Ice','Ship']
TestDir="\\IceData\\test_imgs\\"
TrainedWeights=ROOT_DIR+"\\mask_rcnn_iceshiptf1config_0050.h5"
DEFAULT_LOGS_DIR = ROOT_DIR+"\\.logs"
_rawforceData=ROOT_DIR+"\\.rawForcedata"
_rawvidData=ROOT_DIR+"\\.rawVideoData"
_processedData=ROOT_DIR+'\\'+'.processedData'
regex="\d{2,3}p?\d?m_\dths_\d+p\d+kts_\d?p?\dm_\ddeg_\d{3}"
troubleshooting=False

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


def detect(model,frame_num,videopath): 
    cap = cv2.VideoCapture(_rawvidData+'\\'+videopath)
    # cap=cv2.cvtColor(cap, cv2.COLOR_BGR2RGB)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    res, frame_bgr = cap.read()
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # Perform a forward pass of the network to obtain the results
    r = model.detect([frame], verbose=0)
    r = r[0]
    
    if troubleshooting:
        visualize(frame,r) 
        
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
        t=df[:1].values.tolist()[0]
        
    except: 
        print("Could not open:",pth,"\nFull path:",csvpath)
      
    return df
    
def validateSize(videoDir, ForceDir): 
    #this fcn validates that the same number of frames are in both the 
    # Step 1: Check if the same files are present in the two dirs
    # Step 2: Check if the are recorded as the same number of frames
    
    videoDirWorking=videoDir.copy() #if you dont copy it this way remove() will remove the item from both lists later (they still remain connected if you just set them equal)
    #the regex format is the regex of NRC OCRE file format naming conventions
    
    #step1 - do we have the same files in the same dirs?
    forceDir_reMatch = [re.match(regex, f).group() for f in ForceDir if re.match(regex, f)]
    viddir_reMatch = [re.match(regex, v).group() for v in videoDirWorking if re.match(regex, v)]
    
    #make sure we did not miss a file if we didnt need to - ideally this is checked manually for typos infilenames after seeing this output.
    if len(forceDir_reMatch) != len(ForceDir): 
        print("Regex dropped a file from the force input files")
    if len(viddir_reMatch) != len(videoDirWorking): 
        print("Regex dropped a file from the video input files")
    
    for fo in forceDir_reMatch: 
        for vi in viddir_reMatch: 
            if fo==vi: 
                forceDir_reMatch.remove(fo)
                
    if len(forceDir_reMatch) == 0: 
        "All expected videos are present."
    
    #step 2 -- ensure the number of frames matches the expected frames from the dataset csv file.
    different=[]
    correspondingfiles={}
    for file in ForceDir: 
        name=(re.match(regex,file)).group()
        for vi in videoDirWorking:
            if name in vi: 
                videofile=vi
                videoDirWorking.remove(vi)
                correspondingfiles[file]=vi
                break                 
                
        if NumFrames(videofile) != numframes_forcedata(file): 
            # print("\n\nNumber of frames do not match!!",name)
            different.append(name)
    
    # print("\n\nDifferences in number of frames actual vs recorded!",different)
    
    # TODO: with the corresponding files dictionary here we could definitely streamline the first stage of this function.....
    return [different,correspondingfiles]

def processDetections(r): 
    # this function will process the results from the detected model (r) then will pass the new out to then be combined
    # and saved into the updated csv format to pass along to the training stage.
    # this is done on a per frame basis
    
    ## general procedure: 
    #   1) In comes "r" the results for the frame being detected.
    #       a) define the region bounds: the radii and the angles
    #       b) make a f'region_{angle1}:{angle2}_{r_min}:{r_max}={} dictionary.
    #           End keys will be: 'detection index' as int, where the value will be a running tally of the area
    #   2) Make sure that there is a ship detected, and only ONE ship detected
    #       a) if zero ships; initialize the new parameters into the results dictionary as empty and move to next frame.
    #   3) get ship centroid, and length (see jupyter notebook - already implemented)
    #   4) For every mask; add the following items to the "r" dictionary
    #       a) centroid of that mask
    #           - r['centroid']=[x,y]
    #   5) Calculate the euclidian (straight line) distance to the centroid of the ship, if the distance < 3 ship lengths then do the following; if not, jump to next detection
    #   **6) Read indecies of the mask belonging to that detection; the indeces of the mask coorelate to the coordinates of each pixel of the image.
    #           For each pixel, calculate the euclidian distance to the ship centroid, and the angle.
    #   **7) Based on the angle and euclidian distance, search trough list of regions (by their defining angles and radius) to determine the region that pixel belongs to
    #   **8) Add that one pixel value to the region it corresponds to's running area tally - literally adding +1 each time. Maybe have a set of IF statements that will direct you to the region you fall within.
    #   **9) REPEAT FOR ALL DETECTIONS
    # 
    #   10) calculate the ice concentration in each region, (will need to sum area from each detection region that is contained in that region dict)
    #       Calculate the ice floe size distribution (for this we will need the area of each floe contained in the region; hence why above struct.)
    # 
    
    # Open variables of results that we will use later       
    masks=r['masks']
    classes=r['class_ids']
    
    # Get the indecies of "ship" detections (predictions)
    shipIndex=np.where(classes==CLASS_NAMES.index('Ship'))[0]
    
    # Eventually make this a method to consider the ship with the highest probability
    # within a certain expected region; "the" ship that we use from here out
    if shipIndex.shape[0] != 1:
        ## TODO: Implement the notes above, for now skip this entire frame.
        print("\n\n we have identified more than one ship \n\n")
        return  "something here" ## add somethign here that exits out of the frame
    
    # Assuming we only have one ship detected now (not zero not > 1)
    # Get ship centroid & length
    o=getCentroid(masks[:, :, shipIndex[0]][:, :, np.newaxis],GetShipLength=True) #returns x,y and ship length
    ShipCentroid=o[0:2]
    ShipLength=o[2]
    
    # Define region bounds
    RegionDefinition={"Radii":[0.5*ShipLength,1*ShipLength,2.5*ShipLength],"AngleIncrements":[20,60,135,180]} #do NOT explicitly say 0 degrees.
    
    # Initialize dictionary of the regions; these will be filled in later with sub-dictionaries that have the "index-class" of the detection we found
    regionStats={}
    prevRadi=0
    prevAngl=0
    regionStats["Centroids"]=[] #empty list, eventually will be a list of lists, where each sub list is the x,y coords of a centroid. Purely for vis purposes later
    for Radi in RegionDefinition["Radii"]:
        for angl in RegionDefinition["AngleIncrements"]: 
            regionStats[f"Region_{prevRadi}:{Radi}_{prevAngl}:{angl}"]={}
            if prevAngl != 0:
                regionStats[f"Region_{prevRadi}:{Radi}_{-prevAngl}:{-angl}"]={}
            else: 
                regionStats[f"Region_{prevRadi}:{Radi}_{prevAngl}:{-angl}"]={}
            prevAngl=angl
        prevRadi=Radi
        prevAngl=0
    
    # Iterate over every prediction detection
    for N0 in range(0,classes.shape[0]): 
        t0=time.time()
        centroid=getCentroid(masks[:, :, N0][:, :, np.newaxis])
        regionStats["Centroids"].append(centroid)
        Dist2ship=math.sqrt(((ShipCentroid[0]-centroid[0])**2) + ((ShipCentroid[1]-centroid[1])**2)) #euclidian distance b/w ship centroid and detection centroid
        
        #only do these things if the centroid of this detection is < threshold away from the ship centroid -- limit is purely for computational efficiency
        if Dist2ship < 3.5 * ShipLength :
            
            # Iterate over every pixel in the current mask (that has centroid < 3.5 SL away) and assign +1 to the area running sum for 
            # the region it falls within
            current_mask=masks[:, :, N0][:, :, np.newaxis]

            #Get indeces of mask pixels that evaluate to True, store x and y indecies in cx and cy vars
            Cx=np.where(current_mask)[1] ## C[0] is y axis and C[1] is x axis!!!!!!!!!!!!!! pulls in bkwrds!!!
            Cy=np.where(current_mask)[0] ## C[0] is y axis and C[1] is x axis!!!!!!!!!!!!!! pulls in bkwrds!!!
            
            for item in range(0,len(Cx)): 
                ccord=[Cx[item],Cy[item]]
                R=math.sqrt(((ShipCentroid[0]-ccord[0])**2) + ((ShipCentroid[1]-ccord[1])**2))
                theta=math.degrees(math.atan2(ShipCentroid[1]-ccord[1],ShipCentroid[0]-ccord[0])) #these angles DO work with the coordinate system we have
                
                rlist=RegionDefinition["Radii"].copy()
                rlist.append(0)
                rbounds=BinarySearch(sorted(rlist),R)
                a=[[value, -value] for value in RegionDefinition["AngleIncrements"]]
                a1=[item for sublist in a for item in sublist] #flatten list of lists
                a1.append(0)
                angleList=sorted(a1)
                abounds=BinarySearch(angleList,theta)

                if rbounds == [0,0] or abounds == [0,0]: 
                    # print("out of range of consideration")
                    None
                # elif rbounds != 0 and abounds != 0: 
                else:
                    # print(R,"dist between",rbounds[0],"&",rbounds[1])
                    # print(theta,"deg between",abounds[0],"&",abounds[1]) 
                    
                    regionStats[f"Region_{rbounds[0]}:{rbounds[1]}_{abounds[0]}:{abounds[1]}"][f'Index:{N0}']=400
                    if not f'Index:{N0}' in regionStats[f"Region_{rbounds[0]}:{rbounds[1]}_{abounds[0]}:{abounds[1]}"]: #initialize: set this index =1
                        regionStats[f"Region_{rbounds[0]}:{rbounds[1]}_{abounds[0]}:{abounds[1]}"][f'Index:{N0}']=1
                    else:
                        regionStats[f"Region_{rbounds[0]}:{rbounds[1]}_{abounds[0]}:{abounds[1]}"][f'Index:{N0}']+=1
                
        print("Mask Number:",N0," took:",time.time()-t0," seconds") #takes about 3s per mask so dending on the number of masks in the region
    return [regionStats, RegionDefinition]

def BinarySearch(searchlist,R): 
    indi=bisect.bisect_left(searchlist,R)
    ii=bisect.bisect_right(searchlist,R)
    listlength=len(searchlist)

    if indi == ii: 
        if ii >= listlength and indi >= listlength: # this deals with the case when the number is greater than the max number on the list
            # print("none")
            return [0,0]
        else: #this deals with all 'regular' cases; where the input number is between two items on the list
            # print("regular between:",searchlist[indi-1],searchlist[ii])
            bounds=[searchlist[indi-1],searchlist[ii]]
    if indi != ii:
        if ii >= listlength: # this deals with the case where we are on the upper line - goes into the smaller category
            # print("onthe topline between:",searchlist[indi-1],searchlist[ii-1])
            bounds=[searchlist[indi-1],searchlist[ii-1]]
        else: #this deals with cases where R lies on the dividing line -- goes into the bigger category
            # print("ontheline between:",searchlist[indi],searchlist[ii])
            bounds=[searchlist[indi],searchlist[ii]] 
    
    #we want to read bounds as 0:20, and 0:-20 NOT -20:0 (ascending order)
    if bounds[0] >=0 and bounds[1] >=0: #no negatives? pass, it should be in ascending order
        return bounds
    else: # if theres negatrives, make the order descending
        return [bounds[1],bounds[0]]

def getCentroid(mask,GetShipLength=False):
    #m is one mask
    ##get xy coordinates of each item, then avg them
    horiz = np.where(np.any(mask, axis=0))[0]
    verti = np.where(np.any(mask, axis=1))[0]
    
    hori_mean=np.mean(horiz)
    verti_mean=np.mean(verti)
    
    output=[hori_mean,verti_mean]
    
    if GetShipLength:
        #this gets length of the mask- can be used to get shiplength with zero yaw
        hmax,hmin = horiz[[0, -1]]
        ShipL=abs(hmax-hmin) #shiplength in px
        output.append(ShipL)

    return output

def getShipDrxn(mask,centroid,radius): 
    ## TODO add way to consider alternat headings here
    # this needs to return the DELTA ONLY
    p0=centroid
    ptend=[centroid[0]-radius,centroid[1]] #this returns a point, the distance of 1 radius away
    p1=[-radius,0] #this needs to be updated to account for variations in heading of the masked ship
    
    return [p0,p1,ptend]


def save_newCSV(OriginalData, processedData,filename):
    # this is done on a per frame basis
    filepath=_processedData+'\\'+filename+'.csv'
    
    # Check if the csv file exists yet?
    if not os.path.exists(filepath): #if it doesnt exist yet then put in the basics
        df=pd.DataFrame()
    else: 
        df=pd.read_csv(filepath) 
    
    # prepare new data to be added.
    newrowdata=OriginalData+processedData
    [instance, Carriage_Speed_DP, Global_FX, Global_FY, Global_FT, Global_MZ, Floe_Size, Ice_Conc, Drift_Speed, Ice_Thick, Drift_Angle, Trial,ConcBw_30_azimuth,ConcBw_90_azimuth,ConcBw_180_azimuth]=newrowdata
    newrow_dic={'Instance':instance, 'Carriage_Speed_DP':Carriage_Speed_DP, 'Global_FX':Global_FX, 'Global_FY':Global_FY, 'Global_FT':Global_FT, 'Global_MZ':Global_MZ, 'Floe_Size':Floe_Size, 'Ice_Conc':Ice_Conc, 'Drift_Speed':Drift_Speed, 'Ice_Thick':Ice_Thick, 'Drift_Angle':Drift_Angle, 'Trial':Trial,'ConcBw_30_azimuth':ConcBw_30_azimuth,'ConcBw_90_azimuth':ConcBw_90_azimuth,'ConcBw_180_azimuth':ConcBw_180_azimuth}
    
    # Now we can add the original data, and new data; frame by frame (this fcn gets called each frame)
    new=df.append(newrow_dic,ignore_index=True)
    new.to_csv(filepath) #saves the new file each round incase there is an error we dont want it to be living in memory.

def PostProcess_OriginalData(OriginalData):
    #Format for the original data input list;
    # OriginalData = [FrameN (int),Carriage_Speed_DP (float),Global_FX (float),Global_FY (float),Global_FT (float), ...
    #                   Global_MZ (float),Floe_Size (string), Ice_Conc (string),Drift_Speed (string), ...
    #                   Ice_Thick (string),Drift_Angle (string),Trial (int)]
    
    # In theory this function will be used to calculate the instantaneous speed of the vessel rather than relying on the current
    # speed data that we have access to in the NRC dataset; being carriage speed; which only roughly approximates the speed of
    # of the vessel which is under DP control.
    
    # These functions just make sure we have numerical values rather than strings that we save.
    floesize=float(OriginalData[6].replace('p','.'))
    driftspeed=float(OriginalData[8].replace('kts','').replace('p','.'))
    iceThickness=float(OriginalData[9].replace('m','').replace('p',''))

    # Here grab only the relevant data that we want to feed into the next model -- everything except Ice_Conc, Trial 
    ProcessedOriginalData=OriginalData[0:6]+OriginalData[8:11] #this needs to be a list of things in the following order: 
    # ProcessedOriginalData = [FrameN (int),Carriage_Speed_DP (float),Global_FX (float),Global_FY (float),Global_FT (float), ...
    #                   Global_MZ (float),Floe_Size (string),Drift_Speed (float), ...
    #                   Ice_Thick (float),Drift_Angle (float)]
    #
    ## Realize that Ice_Conc and trial number are removed.
    ProcessedOriginalData = OriginalData[0:6]+[floesize]+[driftspeed]+[iceThickness]+[OriginalData[10]]
    return ProcessedOriginalData

def convertRegionStats (regionStats,RegionDefinition):
    prevRadi=0
    prevAngl=0
    regionIceFinalStats={}
    regionIceFinalStats["RegionNames"]=[]
    regionIceFinalStats["IceConcentratons_pct"]=[]
    regionIceFinalStats["StandardDeviation"]=[]
    for Radi in RegionDefinition["Radii"]:
        for angl in RegionDefinition["AngleIncrements"]: 
            # print("radi range",prevRadi,":",Radi)
            # print("angle range",prevAngl,":",angl)
            if len(regionStats[f"Region_{prevRadi}:{Radi}_{prevAngl}:{angl}"].values()) > 0:
                TotalArea=((angl-prevAngl)/360)*math.pi*((Radi**2)-(prevRadi**2))
                # print(TotalArea, "\n\n")
                IceArea=sum(regionStats[f"Region_{prevRadi}:{Radi}_{prevAngl}:{angl}"].values())
                IceConcentration=IceArea/TotalArea
                StdDev=statistics.pstdev(list(regionStats[f"Region_{prevRadi}:{Radi}_{prevAngl}:{angl}"].values()))
                # print(regionStats[f"Region_{prevRadi}:{Radi}_{prevAngl}:{angl}"])
                # print(sum(regionStats[f"Region_{prevRadi}:{Radi}_{prevAngl}:{angl}"].values()))
                regionIceFinalStats["RegionNames"].append(f"Region_{prevRadi}:{Radi}_{prevAngl}:{angl}")
                regionIceFinalStats["IceConcentratons_pct"].append(IceConcentration)
                regionIceFinalStats["StandardDeviation"].append(StdDev)
            else: 
                print("nolength")
                regionIceFinalStats["RegionNames"].append(f"Region_{prevRadi}:{Radi}_{prevAngl}:{angl}")
                regionIceFinalStats["IceConcentratons_pct"].append(0)
                ["StandardDeviation"].append(0)
                
            prevAngl=angl
        prevRadi=Radi
        prevAngl=0
    
    return regionIceFinalStats 
 
def troubleshootdetection(model):
    #this is hard coding the image that we want to use for devel. purposes.
    image = cv2.imread(r"C:\Users\logan\Desktop\MEng\Mask_RCNN\IceData\test_imgs\100m_dist_9ths_1p2kts_0p4m_0deg_001_c_overhead_frame361.png") #picks a random image in the kangaroo test image dir.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r = model.detect([image], verbose=0)
    r = r[0]
    return r,image


def addlines(plt,radius,angleIncrement,center,equalangles=True):
    ##if equalangles is true then angle increment should be an integer, otherwise
    angles=[]
    if equalangles: 
        if isinstance(angleIncrement, float):
            print("angle increment should be integer")
            exit()
        for n in range(1,round(360/angleIncrement)):
            angles.append(n*math.radians(angleIncrement))
    
    if not equalangles:
        for deg in angleIncrement: 
            angles.append(math.radians(deg))
            angles.append(math.radians(-deg))
    
    for angle in angles: 
        plt.plot([center[0],center[0]-radius*math.cos(angle)],[center[1],center[1]+radius*math.sin(angle)])


def viz_centroids(image,centroidlist,r,ShowCentroids=True,ShowRegions=True):
    import matplotlib.pyplot as plt
    import pylab
    plt.imshow(image)
    
    masks=r['masks']
    classes=r['class_ids']
    
    shipIndex=np.where(classes==CLASS_NAMES.index('Ship'))[0]
    shipx,shipy,ShipLength=getCentroid(masks[:, :, shipIndex[0]][:, :, np.newaxis],GetShipLength=True)
    
    if ShowCentroids:
        n=0
        for i in centroidlist: 
            plt.plot(i[0],i[1],'rx',markersize=5)
            pylab.text(i[0]+30,i[1],n)
            n+=1
    
    if ShowRegions:
        radii=[0.5,1,2.5]
        for radius in radii:
            plt.gca().add_patch(plt.Circle((shipx,shipy), radius*ShipLength, color='black', fill=False))

        addlines(plt,radius=2.5*ShipLength,equalangles=False,angleIncrement=[0,20,60,135,180],center=[shipx,shipy])
        
    plt.xlim(0, image.shape[1])
    plt.ylim(0, image.shape[0])
    plt.show()
    
    plt.show()

if __name__ == "__main__": 
    config=loadconfig()
    config.display()
    videoDir, ForceDir=listbothdirs()
    diff,correspondingfiles=validateSize(videoDir, ForceDir)
    if len(diff) != 0: 
        print("Look into these differences",diff)
    ## up to here is working; in the test dataset - the videos and force containing csv files all have the same number of frames recorded.
    
    ## TODO (WIP): Now add iterative method to grab each datafile (CSV), extract the data we want to train a model on then rebuild a csv file for that video instance.
    mdl=init_model(config)
    for forcefilename in correspondingfiles: 
        baseFilename=re.match(regex, forcefilename).group() #this extracts the important stuff not the garbage extras that are incl.
        videofilename=correspondingfiles[forcefilename]
        OriginalData=GetCSVData(forcefilename) #returns a pandas dataframe.
        for frameN in range(0,OriginalData.iat[-1,0]): #for range 0-number of frames (contained in cell )
            # r=detect(mdl,frameN,videofilename)
            r,image=troubleshootdetection(mdl) ## only use this for troubleshooting; remove later.
            regionStats,regiondefs =processDetections(r)
            viz_centroids(image,regionStats['Centroids'],r)
            
            # Concert regionStats dict of dicts into ice concentration and ice size distribution data for each region;
            convertRegionStats(regionStats,regiondefs) #TODO: finsih this and merge it with the processDetections function.
            
            # From Shameem: the speed recorded in these preliminary datafiles are the carriage speed; while the vessel is under DP control.
            # We need to consider the thrust --> speed conversion here to get the actual vessel speed.
            ProcessedOriginalData=PostProcess_OriginalData(OriginalData[:frameN+1].values.tolist()[0])

            #save a csv file for every frame - overwriting the previous so we can pickup where we left off.
            save_newCSV(ProcessedOriginalData,regionStats,baseFilename) #not allowed to grab the column names as the first row; needs to be row 1
        

    
