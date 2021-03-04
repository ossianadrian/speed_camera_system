#write all global variables here

# representing pixels per meter to calculate speed
# PPM = 3.4
global PPM
PPM = 45
# constant for framerate of the video. Autoinitialized
global FPS
FPS = 0
# used for conversion from m/s in km/h
global MS_KMPH
MS_KMPH = 3.6
# dimensions for video resizing
global HEIGHT
global WIDTH
HEIGHT = 720
WIDTH = 1280
# 2 points in the image, representing a line which a vehicle passed and camera registers the speed
global START_POINT
global START_POINT_LIST
global END_POINT
global END_POINT_LIST
START_POINT = (57, 488)
START_POINT_LIST = [57, 488]
END_POINT = (476, 470)
END_POINT_LIST = [476, 470]

# variables representing the parameters of the line ecuation for the points above
global A, B, C
A = 0
B = 0
C = 0

# 2 points in the image, representing the lower line which a vehicle passed and camera registers the speed
global START_POINT2
global START_POINT_LIST2
global END_POINT2
global END_POINT_LIST2

START_POINT2 = (154, 626)
START_POINT_LIST2 = [154, 626]
END_POINT2 = (915, 549)
END_POINT_LIST2 = [915, 549]

# variables representing the parameters of the line ecuation that is located lower for the points above
global A2, B2, C2
A2 = 0
B2 = 0
C2 = 0

# the distance between the 2 lines in real life
global distanceBetweenThe2Lines
distanceBetweenThe2Lines = 15.75

# constant for getting relation between 720p image and 4k image

IMG_720p_TO_2160p = 2.986111111

#for yoloV3

global whT
whT = 320

#confidence threshhold
global confThreshHold
confThreshHold = 0.5

# the higher, the agressive and less number of bounding boxes
global nmsThreshHold
nmsThreshHold = 0.6

global classesFile
classesFile = 'coco.names'
global classNames
classNames = []
global net 
