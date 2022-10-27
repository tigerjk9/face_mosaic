import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # 얼굴을 찾기 위한 알고리즘이 적용된 파일 불러오기
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')  # 눈을 찾기 위한 알고리즘이 적용된 파일 불러오기

ff = np.fromfile(r'23. 사진에서 얼굴만 찾아 모자이크처리 (OpenCV)\샘플사진.jpg', np.uint8) # opencv에서 한글경로 파일을 읽지 못해 numpy로 파일을 읽어옵니다.
img = cv2.imdecode(ff, cv2.IMREAD_UNCHANGED) # imdecode 하여 numpy 이미지 파일을 OpenCV 이미지로 불러옵니다.
img = cv2.resize(img, dsize=(0, 0), fx=1.0, fy=1.0, interpolation=cv2.INTER_LINEAR) # 이미지 크기를 조절합니다. fx, fy의 비율로 조절가능하며 코드에서는 원래의 비율로 사용합니다.

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 이미지에서 얼굴을 찾기 위해 회색조 처리합니다.

faces = face_cascade.detectMultiScale(gray, 1.2,5) # 여러 개의 얼굴을 찾습니다. 1.2는 스케일팩터(감도), 5는 minNeighbor(최소 이격 거리)을 나타냅니다.  
for (x,y,w,h) in faces:  # 얼굴을 찾아 파란색으로 네모 표시를 합니다. 
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0),2)

    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:    # 눈을 찾아 녹색 네모 표시를 합니다.
        cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh),(0,255,0),2)

cv2.imshow('face find', img)
cv2.waitKey(0)
cv2.destroyAllWindows()