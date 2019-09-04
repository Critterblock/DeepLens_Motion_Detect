#The code below is a DeepLens adaptation of an existing motion detection script by Adrian Rosebrock at pyimagesearch.com
#For more on this approach, check out these pages: https://bit.ly/2C8edUi and https://bit.ly/2HUIid2
#Special thanks to Thorsten Graf for helping to refine this code

#std library
from datetime import datetime
import numpy
import time
import os

#native to awscam
import awscam

#pip installed
import imutils
import cv2

#Wait for SD Card
#time.sleep(180) #Uncomment if you initiate the code at system startup and want to wait for sd-card

#configs section
debug_level = 0     # 0-save only pics; 1-save debug frames; 2-save debug frames even without motion
color_threshold = 40 #pixel diff threshold to turn image into BW silouhette showing changes
area_threshold = 300 #area threshold for the size of contour that triggers motion detection
blur = 25 # Gaussian blur level. NOTE- THIS MUST BE AN ODD NUMBER
avg_input_weight = 0.3

# /share/Images/ - 4000 limit
# /media/aws_cam/3235-3034/camrecords/  - 7000 limit
saveDir = '/media/aws_cam/7CFB-7F00/' #Location of where images are saved. Your location will vary, so configure this. Recommend clearing every 24 hours with S3 upload
FileLimit = 6000 #If we save too many files, stop the script to keep the hard drive from filling completely.

# initialize variables
avg = None
TotalSaves = 0 #how many total images have we saved onto the hard disk.

def create_average(frame): #take a frame and create a running average frame
    frame = frame[650:1520, 1200:1550] #Adjust this so that it crops to your cat's approach vector
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Turn to greyscale
    background = cv2.GaussianBlur(frame, (blur, blur), 0)
    print("[INFO] starting background model...")
    avg = background.copy().astype("float")
    return(avg)

# If there's no average, initialize it.
ret, frame = awscam.getLastFrame() #turn this into a get&crop&grey function
if avg is None:
    avg = create_average(frame)
    #timestamp = str(datetime.now().strftime("%Y%m%d-%H-%M-%S.%f"))
    #cv2.imwrite(saveDir + timestamp + "_average.jpeg", avg)
    print 'initializing average at ' + datetime.now().strftime("%Y%m%d-%H-%M-%S.%f")

while True:
    ret, frame = awscam.getLastFrame() #turn this into a get&crop&grey function
    if ret == True:
        frame = frame[650:1520, 1200:1550]
        cframe = frame  # store the colored frame as cframe to be able to save it
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Grayscale the image
        gray = cv2.GaussianBlur(frame, (blur, blur), 0)  # blur the image

        # accumulate the weighted average between the current frame and
        # previous frames, then compute the difference between the current
        # frame and running average
        cv2.accumulateWeighted(gray, avg, avg_input_weight)
        frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

        # threshold the delta image, dilate the thresholded image to fill
        # in holes, then find contours on thresholded image
        thresh = cv2.threshold(frameDelta, color_threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        movement = False

        if debug_level == 2:
            timestamp = str(datetime.now().strftime("%Y%m%d-%H-%M-%S.%f"))
            cv2.imwrite(saveDir + timestamp + "_gray.jpeg", gray)
            cv2.imwrite(saveDir + timestamp + "_delta.jpeg", frameDelta)
            cv2.imwrite(saveDir + timestamp + "_thresh.jpeg", thresh)  # Uncomment this if you want to save the tB&W threshold image.)
            cv2.imwrite(saveDir + timestamp + "_average.jpeg", avg)

        for c in cnts:
            area = cv2.contourArea(c)
            if area >= area_threshold:  # if a large enough contour is discovered, return a positive result for movement
                movement = True
                break

        # If movement detected, save image (and colored image and B&W delta image, threshold image and running average image if you want)
        timestamp = str(datetime.now().strftime("%Y%m%d-%H-%M-%S.%f"))
        if movement == True:
            print 'Movement at ' + datetime.now().strftime("%Y%m%d-%H-%M-%S.%f")
            cv2.imwrite(saveDir + timestamp + "_color.jpeg", cframe)
            # cv2.imwrite(saveDir + timestamp + ".jpeg", frame)
            if debug_level == 1:
                cv2.imwrite(saveDir + timestamp + "_gray.jpeg", gray)
                cv2.imwrite(saveDir + timestamp + "_delta.jpeg", frameDelta)
                cv2.imwrite(saveDir + timestamp + "_thresh.jpeg", thresh) #Uncomment this if you want to save the tB&W threshold image.)
                cv2.imwrite(saveDir + timestamp + "_average.jpeg", avg)

        #If we've saved too many files and are filling up the hard drive, kill the script
        TotalSaves = len(os.listdir(saveDir))
        if TotalSaves > FileLimit:
            print 'file limit reached'
            break
