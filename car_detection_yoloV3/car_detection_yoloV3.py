import cv2
import numpy as np


cap = cv2.VideoCapture('test2.mp4')

whT = 320

classesFile = 'coco.names'
classNames = []

#confidence threshhold
confThreshHold = 0.5

# the higher, the agressive and less number of bounding boxes
nmsThreshHold = 0.6

def findObjects(outputs, img, classNames):
    hT, wT, cT = img.shape
    # list with x,y,w,h
    bbox = []
    classIds = []
    # confidence values
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshHold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int(det[0] * wT - w/2), int(det[1] * hT - h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshHold, nmsThreshHold)

    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%' , (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,255,0), 2)


def trackCars():
    # read file with classes that yolov3 recognizes
    with open(classesFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    # read the conf file and weights of yolov3
    modelConfiguration = 'yolov3.cfg'
    modelWeights = 'yolov3.weights'

    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    

    while True:
        success, img = cap.read()


        # convert image to square to avoi error: (-215:Assertion failed) !ssize.empty() in function 'resize'
        crop_img = img[0:0+1080, 0:0+1080]
        # convert image in a format that yolov3 understands aka blob

        blob = cv2.dnn.blobFromImage(crop_img, 1/255, (whT, whT), [0,0,0], 1, crop = False)
        net.setInput(blob)

        layerNames = net.getLayerNames()
        # print(layerNames)
        net.getUnconnectedOutLayers()
        outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]

        # get the output of the 3 layers. Outputs is a list

        outputs = net.forward(outputNames)

        findObjects(outputs, crop_img, classNames)


        cv2.imshow('Image', crop_img)

        cv2.waitKey(1)

        


if __name__ == '__main__':
    trackCars()

