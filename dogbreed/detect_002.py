import numpy as np
import cv2
###file:///C:/Users/i010747/Downloads/temp/softwares/opencv/opencv/build/etc/haarcascades/haarcascade_eye.x

face_cascade = cv2.CascadeClassifier('C:/Users/i010747/Downloads/temp/softwares/opencv/opencv/build/etc/haarcascades/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('C:/Users/i010747/Downloads/temp/softwares/opencv/opencv/build/etc/haarcascades/haarcascade_eye.xml')

img = cv2.imread('001.jpg')

for low_range in range(1,20):

    canny_img = cv2.Canny(img, low_range * 25, low_range * 25 + 25)
    CANNY_WINDOW_NAME = "canny"
    cv2.namedWindow(CANNY_WINDOW_NAME)
    cv2.imshow(CANNY_WINDOW_NAME, canny_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
#    for (ex,ey,ew,eh) in eyes:
#        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

