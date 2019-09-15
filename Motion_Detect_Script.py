'''
The code below is a DeepLens adaptation of an existing motion detection script by Adrian Rosebrock
at pyimagesearch.com For more on this approach, check out these pages:
https://bit.ly/2C8edUi and https://bit.ly/2HUIid2
Special thanks to Thorsten Graf and Eddie Calleja for helping to refine this code
'''
from datetime import datetime
import os

import awscam

import imutils
import cv2

#configs section
DEBUG_LEVEL = 0 # 0-save only pics; 1-save debug frames; 2-save debug frames even without motion
COLOR_THRESHOLD = 40 #pixel diff threshold to turn image into BW silouhette showing changes against background
AREA_THRESHOLD = 300 #area threshold for the size of contour that triggers motion detection
BLUR = 25 # Gaussian BLUR level. NOTE- THIS MUST BE AN ODD NUMBER
AVG_INPUT_WEIGHT = 0.3

#Get the name of the SDCard in your DeepLens (it comes in the box, so make sure you put it in!)
MEDIA_PATH = '/media/aws_cam/'
SDCARD_NAME = [item for item in os.listdir(MEDIA_PATH) if len(item) == 9 and "-" in item][0]
SDCARD_PATH = MEDIA_PATH + SDCARD_NAME

#Location of where images are saved. Your location will vary, so configure this. Recommend
# clearing every 24 hours with S3 upload
SAVE_DIR = SDCARD_PATH #You can set this to on-device storage, but that's not a great idea.

#If we save too many files, stop the script to keep the hard drive from filling completely.
FILE_LIMIT = 20000 #Vary this depending on the size of your images

def create_average(frame):
    ''' Method for creating a running average frame
        frame - input frame for creating the running average frame
    '''
    #Adjust frame shape to crop what you to capture (saves lots of disk space)
    #frame = frame[650:1520, 1200:1550]
    #Turn to greyscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(frame, (BLUR, BLUR), 0)
    print("[INFO] starting background model...")
    avg = background.copy().astype("float")
    return avg

def dump_images(save_dir, time_stamp, gray, frame_delta, thresh, avg):
    ''' Helper method that dumps allthe images used to determine motion. Intended
        for debugging purposes
    '''
    cv2.imwrite(os.path.join(save_dir, '{}_{}'.format(time_stamp, 'gray.jpeg')), gray)
    cv2.imwrite(os.path.join(save_dir, '{}_{}'.format(time_stamp, 'delta.jpeg')), frame_delta)
    cv2.imwrite(os.path.join(save_dir, '{}_{}'.format(time_stamp, 'thresh.jpeg')), thresh)
    cv2.imwrite(os.path.join(save_dir, '{}_{}'.format(time_stamp, 'average.jpeg')), avg)

def detect_motion():
    ''' Method for detecting motion and saving the images with detected motion
    '''
    print('initializing average at {}'.format(datetime.now().strftime("%Y%m%d-%H-%M-%S.%f")))
    ret, frame = awscam.getLastFrame() #bootstrap the running average frame
    avg = create_average(frame)

    while True:
        ret, frame = awscam.getLastFrame()
        if ret:
            frame = frame[650:1520, 1200:1550]
            # store the colored frame as cframe to be able to save it
            cframe = frame
            # Grayscale the image
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # BLUR the imageGaussianBLUR
            gray = cv2.GaussianBlur(frame, (BLUR, BLUR), 0)

            # accumulate the weighted average between the current frame and
            # previous frames, then compute the difference between the current
            # frame and running average
            cv2.accumulateWeighted(gray, avg, AVG_INPUT_WEIGHT)
            frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

            # threshold the delta image, dilate the thresholded image to fill
            # in holes, then find contours on thresholded image
            thresh = cv2.threshold(frame_delta, COLOR_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            movement = False

            if DEBUG_LEVEL == 2:
                time_stamp = str(datetime.now().strftime("%Y%m%d-%H-%M-%S.%f"))
                dump_images(SAVE_DIR, time_stamp, gray, frame_delta, thresh, avg)

            for count in cnts:
                area = cv2.contourArea(count)
                # if a large enough contour is discovered, return a positive result for movement
                if area >= AREA_THRESHOLD:
                    movement = True
                    break

            # If movement detected, save image (and colored image and B&W delta image, threshold
            # image and running average image if you want)
            time_stamp = str(datetime.now().strftime("%Y%m%d-%H-%M-%S.%f"))
            if movement:
                print('Movement at {}'.format(datetime.now().strftime("%Y%m%d-%H-%M-%S.%f")))
                cv2.imwrite(os.path.join(SAVE_DIR, '{}_{}'.format(time_stamp, 'color.jpeg')),
                            cframe)
                if DEBUG_LEVEL == 1:
                    dump_images(SAVE_DIR, time_stamp, gray, frame_delta, thresh, avg)

            #If we've saved too many files and are filling up the hard drive, kill the script
            if len(os.listdir(SAVE_DIR)) > FILE_LIMIT:
                print('file limit reached')
                break

if __name__ == "__main__":
    detect_motion()
