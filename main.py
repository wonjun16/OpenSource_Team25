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

face_cascade=cv2.CascadeClassifier('./haarcascade_file/haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('./haarcascade_file/haarcascade_eye.xml')
mouth_cascade=cv2.CascadeClassifier('./haarcascade_file/haarcascade_smile.xml')

while True : 
    
    image = input("Select image >> Einstein : e / Monalisa : m / Solbay : s / Exit : x\n")
    
    if image not in {'e', 'm', 's', 'x'} :
        print("This is not valueable word. Please put another word.")
        continue
    elif image=='x' :
        break
    elif image=='e' : 
        src = cv2.imread('./image/einstein.jpg')
    elif image=='m' :
        src = cv2.imread('./image/monalisa.jpg')
    elif image=='s' :
        src = cv2.imread('./image/solbay.jpg')
        
    dst_area = src
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    
    while True : 
        
        action = input("Select action >> Detect face : d / Mosiac face : m / Detext eyes : e / Mosiac eyes : y / Exit : x\n")
        
        if action not in {'d', 'm', 'e', 'y', 'x'} :
            print("This is not valueable word. Please put another word.")
            continue
        elif action=='x' : 
            break
        elif action=='d' :
            
            faces=face_cascade.detectMultiScale(src_gray, 1.1, 4)
            
            for(x, y, w, h) in faces:
                cv2.rectangle(src, (x, y), (x+w, y+h), (20, 20, 20), 2)
                
            cv2.imshow('img', src)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            continue
                
        elif action=='m' :
            
            for x, y, w, h in faces:
                dst_area = mosaic_area(dst_area, x, y, w, h)
                
            cv2.imshow('img', dst_area)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            continue
        
        elif action=='e' :
            
            faces = face_cascade.detectMultiScale(src_gray,1.1,4)
            
            for x, y, w, h in faces:
                face = src[y: y + h, x: x + w]
                face_gray = src_gray[y: y + h, x: x + w]
            
            eyes = eye_cascade.detectMultiScale(face_gray)
            
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(face, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                
            cv2.imshow('img', src)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            continue
        
        elif action=='y' :
            
            faces = face_cascade.detectMultiScale(src_gray,1.1,4)
            
            for x, y, w, h in faces:
                face_gray = src_gray[y: y + h, x: x + w]
            
            eyes = eye_cascade.detectMultiScale(face_gray)
            
            for (ex, ey, ew, eh) in eyes:
                dst_area = mosaic_area(dst_area, ex, ey, ew, eh)
                
            cv2.imshow('img', dst_area)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            continue