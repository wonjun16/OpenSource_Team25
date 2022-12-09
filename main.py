import cv2

#모자이크 처리
def mosaic(src, ratio=0.1):
    small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

#특정영역 모자이크 처리
def mosaic_area(src, x, y, width, height, ratio=0.1):
    dst = src.copy()
    dst[y:y + height, x:x + width] = mosaic(dst[y:y + height, x:x + width], ratio)
    return dst

#얼굴, 눈 인식 딥러닝파일 경로
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#테스트 이미지
src = cv2.imread('TestImage2.jpg')
dst_area = cv2.imread('TestImage2.jpg')
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

#얼굴 인식
faces = face_cascade.detectMultiScale(src_gray,1.1,3)

#검출된 부분 사각형 그리기
for x, y, w, h in faces:
    print('xywh : ' , x, y, w, h)
    cv2.rectangle(src, (x, y), (x + w, y + h), (255, 0, 0), 2)
    dst_area = mosaic_area(dst_area, x, y, w, h)
    face = src[y: y + h, x: x + w]
    face_gray = src_gray[y: y + h, x: x + w]
    
    #눈 인식
    eyes = eye_cascade.detectMultiScale(face_gray)
    for (ex, ey, ew, eh) in eyes:
        print('ex,ey,ew,eh : ' , ex, ey, ew, eh)
        dst_area = mosaic_area(dst_area, ex, ey, ew, eh)
        cv2.rectangle(face, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

cv2.imwrite('face_mosaic_area.jpg', dst_area)

cv2.imwrite('face_cascade.jpg', src)