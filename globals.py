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
START_POINT = (180, 500)
START_POINT_LIST = [180, 500]
END_POINT = (520, 460)
END_POINT_LIST = [520, 460]

# variables representing the parameters of the line ecuation for the points above
global A, B, C
A = 0
B = 0
C = 0

