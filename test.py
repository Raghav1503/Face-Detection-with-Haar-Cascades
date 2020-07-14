import numpy as np
import cv2 as cv
import urllib.request

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

url='https://192.168.29.12:8080/shot.jpg'

# loading cascades
face_cascade = cv.CascadeClassifier(r'C:\Users\ragha\Desktop\New folder\data\haarcascades\haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(r'C:\Users\ragha\Desktop\New folder\data\haarcascades\haarcascade_eye.xml')
smile_cascade = cv.CascadeClassifier(r'C:\Users\ragha\Desktop\New folder\data\haarcascades\haarcascade_smile.xml')

# detecting features and marking them
def detect(gray, frame):
    # detecting faces in the figure
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # creating a rectangle around the detected face
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # rectangle(image, top-left corner of image, down-right corner of image, color of border, thickness of border)

        # basically cropping image(using rectanglular detected face for further computations)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # detecting eyes in the cropped image
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)

        # creating a rectangle around the detected eyes
        for (a, b, w1, h1) in eyes:
            cv.rectangle(roi_color, (a, b), (a + w1, b + h1), (0, 0, 255), 2)
        smile = smile_cascade.detectMultiScale(roi_gray, 1.1, 22)
        for (p, q, w2, h2) in smile:
            cv.rectangle(roi_color, (p, q), (p+w2, q+h2), (0, 255, 0), 2)
    return frame

#video_capture = cv.VideoCapture(0)
#while True:
 #   _, frame = video_capture.read()
frame = cv.imread('test1.jpg')
#cv.imshow('img',frame)
#cv.waitKey(1)
gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#cv.imshow('img1',gray)
#cv.waitKey(1000)
#canvas = detect(gray, frame)
cv.imshow('Video', canvas)
cv.waitKey(1000)
#    if cv.waitKey(1) & 0xFF == ord('q'):
 #       break
#video_capture.release()
cv.destroyAllWindows()


while True:
    imgResp=urllib.request.urlopen(url)
    imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
    frame=cv.imdecode(imgNp,-1)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)

    # all the opencv processing is done here
    cv.imshow('test',canvas)
    if ord('q')==cv.waitKey(10):
        exit(0)


'''cap = cv.VideoCapture('C:\MOVIES AND TV SHOWS\TV Series\Friends\Season 1\S01E03 the one with the thumb.mkv')

while True:
    #img=cv2.imdecode(frame,-1)
    _, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # all the opencv processing is done here
    canvas = detect(gray, frame)

    # all the opencv processing is done here
    cv.imshow('test',canvas)
    if ord('q')==cv.waitKey(10):
        exit(0)'''



