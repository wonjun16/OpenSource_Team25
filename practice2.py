import cv2

#haarcascade 불러오기
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('haarcascade_eye.xml')
mouth_cascade=cv2.CascadeClassifier('haarcascade_smile.xml')

#이미지 가져오기
img=cv2.imread('./image/monalisa.jpg')

#이미지 전처리
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#얼굴 찾기
faces=face_cascade.detectMultiScale(gray, 1.1, 4)
for(x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (20, 20, 20), 2)
    face_gray=gray[y:y+h, x:x+w]
    face_color=img[y:y+h, x:x+w]
    
    #눈 찾기
    eyes=eye_cascade.detectMultiScale(face_gray, 1.01, 2)
    for(ex, ey, ew, eh) in eyes:
        cv2.rectangle(face_color, (ex, ey), (ex+ew, ey+eh), (200, 20, 20), 1)

    #입 찾기
    mouth=mouth_cascade.detectMultiScale(face_gray, 1.1, 4, minSize=(10,1))
    for(sx, sy, sw, sh) in mouth:
        cv2.rectangle(face_color, (sx, sy), (sw+sw, sy+sh), (40, 0, 40), 2)    

#결과 출력
cv2.imshow('img', img)
key=cv2.waitKey(0)
cv2.destroyAllWindows()