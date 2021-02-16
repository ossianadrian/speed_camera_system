from math import sqrt, pow
from cv2 import CascadeClassifier, VideoCapture, VideoWriter, VideoWriter_fourcc, CAP_PROP_FPS, resize, cvtColor, COLOR_BGR2GRAY, rectangle, putText, waitKey, imshow, imwrite, destroyAllWindows, line
import dlib
import globals

# cascade haaR classifier for cars
CAR_CASCADE_HAAR_CLASSIFIER = CascadeClassifier('car_haar_classifier.xml')
# import video
VIDEO = VideoCapture('test2.mp4')


def calculateSpeed(point1, point2, FPS, PPM):
    # sqrt( (p2.a - p1.a)^2 + (p2.b - p1.b)^2 )
    distance_in_pixels = sqrt(pow(point2[0] - point1[0], 2) + pow(point2[1] - point1[1], 2))
    distance_in_meters = distance_in_pixels / PPM
    # return the speed calculated in km/h
    return distance_in_meters * FPS * globals.MS_KMPH

def calculateLineEcuation(P, Q):
    # the final ecuation will be ax + by = c 
    # stored in globals
    globals.A = Q[1] - P[1]
    globals.B = P[0] - Q[0]
    globals.C = globals.A * (P[0]) + globals.B * (P[1])

# **works only if P is to the left and under the Q
def checkIfPointIsBelowLine(a, b, c, thePoint):

    y_on_line = (c - a * thePoint[0] ) / b
    #check if the y of thePoint is smaller than y of the intersation with the line. If so, the point is below the line
    if thePoint[1] >= y_on_line:
        return True
    else:
        return False

def trackCars():

    # create a file in which to write the output video
    output_video = VideoWriter('outputVideo.avi', VideoWriter_fourcc('M','J','P','G'), 20, (globals.WIDTH, globals.HEIGHT))
    current_car_id = 0
    frame_cnt = 0
    # get the fps of the video
    globals.FPS = VIDEO.get(CAP_PROP_FPS)

    car_tracker_dict = {}
    cars_point1 = {}
    cars_point2 = {}
    speed_of_cars = [None] * 1000
    extracted_cars = [None] * 1000

    # calculate ecuation for the line determined by 2 stable points. Only vehicles passing this line will get the speed registered
    calculateLineEcuation(globals.START_POINT_LIST, globals.END_POINT_LIST)

    while True:

        if waitKey(33) == 27:
            break

        ret, image_from_video = VIDEO.read()
        # Count current frame
        frame_cnt = frame_cnt + 1

        # if video is over, break
        if type(image_from_video) == type(None):
            break

        image_from_video = resize(image_from_video, (globals.WIDTH, globals.HEIGHT))
        modified_image = image_from_video.copy()
        # A list of cars to delete from array. In this list IDs are stored
        cars_to_delete = []

        # Check if any car has left the field of view
        for car in car_tracker_dict.keys():
            #Update the tracking accuracity
            tracker_accuracity = car_tracker_dict[car].update(image_from_video)
            if tracker_accuracity < 7:
                cars_to_delete.append(car)

        for car in cars_to_delete:
            print('[Remove] Deleting car with id = ' + str(car))
            # additional argument None needed in case a car is not present in the tracker. Otherwise we get an exception thrown
            car_tracker_dict.pop(car, None)
            cars_point1.pop(car, None)
            cars_point2.pop(car, None)

        #do all these checks each 10 frames

        if frame_cnt % 10 == 0:
            #convert image to grayscale
            grayscale_image = cvtColor(image_from_video, COLOR_BGR2GRAY)
            # use classifier to detect cars
            cars_detected = CAR_CASCADE_HAAR_CLASSIFIER.detectMultiScale(grayscale_image, 1.1, 13, 18, (24, 24))
            
            for (int32_x, int32_y, int32_w, int32_h) in cars_detected:
                #cast to integer python
                x, y, w, h = int(int32_x), int(int32_y), int(int32_w), int(int32_h)

                # calculate the center of gravity for the rectangle that fits the car
                x_center = x + w * 0.5
                y_center = y + h * 0.5

                matched_car = None
                # detect if car is already found in dictionary
                for car in car_tracker_dict.keys():
                    car_position = car_tracker_dict[car].get_position()

                    tracked_x, tracked_y, tracked_w, tracked_h = int(car_position.left()), int(car_position.top()), int(car_position.width()), int(car_position.height())

                    # calculate the center of gravity for the rectangle in which the tracked car is located
                    tracked_x_center = tracked_x + tracked_w * 0.5
                    tracked_y_center = tracked_y + tracked_h * 0.5

                    # if the center of gravity for the first tracked car is in range of the one calculated from above, then it's a match

                    if (tracked_x <= x_center <= (tracked_x + tracked_w)) and (tracked_y <= y_center <= (tracked_y + tracked_h)) and (x <= tracked_x_center <= (x+w)) and (y <= tracked_y_center <= (y+h)):
                        matched_car = car
                        break

                if matched_car is None:
                    print('[Added car] Creating new tracker for car with id = ' + str(current_car_id))
                    # Create a correlation tracker to track the new identified car in each frame of the video
                    correlation_tracker = dlib.correlation_tracker()
                    correlation_tracker.start_track(image_from_video, dlib.rectangle(x, y, x+w, y+h))
                    car_tracker_dict[current_car_id] = correlation_tracker
                    # Save the first point used for calculating speed
                    cars_point1[current_car_id] = [x, y, w, h]
                    current_car_id = current_car_id + 1

            # do this code for all the cars present in the dictionary
        for car in car_tracker_dict.keys():
            car_position = car_tracker_dict[car].get_position()
            tracked_x, tracked_y, tracked_w, tracked_h = int(car_position.left()), int(car_position.top()), int(car_position.width()), int(car_position.height())

            # draw a rectangle on the image to identify the car visually
            rectangle(modified_image, (tracked_x, tracked_y), (tracked_x + tracked_w, tracked_y + tracked_h), (0, 0 , 255), 2)

            # Add point 2 for this car for speed estimation 
            cars_point2[car] = [tracked_x, tracked_y, tracked_w, tracked_h]


        for i in cars_point1.keys():

            [x1, y1, w1, h1] = cars_point1[i]
            [x2, y2, w2, h2] = cars_point2[i]

            if cars_point1[i] != cars_point2[i]:
                x_center = x1 + w1 * 0.5
                y_center = y1 + h1 * 0.5

                # calculate the speed only if the car passed the point 275 450 and the speed is 0 or None
                # if (speed_of_cars[i] == 0 or speed_of_cars[i] == None) and y1 >= 275 and y1 <= 450:

                # modified_image = line(modified_image, (int(x_center), int(y_center)), END_POINT, (0,0,255), 9)
                if (speed_of_cars[i] == 0 or speed_of_cars[i] == None) and checkIfPointIsBelowLine(globals.A, globals.B, globals.C, [x_center, y_center]) :
                    print('[Info] Calculating speed for car with id = ' + str(i))
                    speed_of_cars[i] = calculateSpeed(cars_point1[i], cars_point2[i], globals.FPS ,globals.PPM)

                if speed_of_cars[i] != None :
                    putText(modified_image, "[" + str(i) + "] " + str(int(speed_of_cars[i])) + "km/h", (int(x1 + w1/2), int(y1-8)), 3 , 0.6 , (255, 255, 255), 2)
                    if extracted_cars[i] == None:
                        print('[LPR] Extracted image for LPR for car with id = ' + str(i) + ' having the speed = ' + str(speed_of_cars[i]) + 'km/h')
                        imwrite('ExtractedImageForLPR' + str(i) + '.png', image_from_video[y2:(y2 + h2), x2:(x2 + w2)])
                        extracted_cars[i] = 1



            cars_point1[i] = cars_point2[i]

        line(modified_image, globals.START_POINT, globals.END_POINT, (0,0,255), 2)
        imshow('modified image', modified_image)

        if waitKey(33) == ord('c'):
            imwrite('testIMG.png', modified_image)

        
        # Write output image to the new video file
        # output_video.write(modified_image)

    destroyAllWindows()









if __name__ == '__main__':
    trackCars()

#calculez ecuatia dreptei

