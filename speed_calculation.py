import cv2


def trackCars():
    cascade_src = 'myhaar.xml'
    video_src = 'test2.mp4'

    video_capture = cv2.VideoCapture(video_src)

    car_cascade = cv2.CascadeClassifier(cascade_src)

    while True:
        ret, img = video_capture.read()

        #if video is done
        if(type(img) == type(None)):
            break

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #objects smaller than 13 and bigger than 24 are discarded
        cars = car_cascade.detectMultiScale(gray_img, 1.1, 13, 18, (24, 24))

        for (x,y,w,h) in cars:
            # print(str(x) + " "+ str(y) + " "+ str(w) + " "+ str(h))
            cv2.rectangle(img, (x,y),(x+w, y+h), (0,0,255), 2)
            # cv2.rectangle(img, (5,5), (1820,1060), (255,0,0),  2)

        cv2.imshow('video', img)

        #if esc key is pressed, abort
        if cv2.waitKey(33) == 27:
            break

    cv2.destroyAllWindows()





if __name__ == '__main__':
    trackCars()
