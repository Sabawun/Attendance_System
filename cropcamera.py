import cv2
cam = cv2.VideoCapture(0)
cam.set(3, 1920) # 640
cam.set(4, 1080) # 480
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
print("\n Initializing...")
count = 0
while(True):
    ret, img = cam.read()
    img = cv2.flip(img, 1)
    faces = face_detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
    for (x,y,w,h) in faces:
        count += 1
        try:
            crop_img = img[y: y+int(h*1.2), x: x+w]
            cv2.imwrite("Onur/Type("+ str(count) + ").jpg", crop_img)
        except:
            crop_img = img[y: y+h, x: x+w]
            cv2.imwrite("Onur/Type("+ str(count) + ").jpg", crop_img)
        cv2.imshow('image', crop_img)
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif count >= 50:
         break
print("\n Exiting Program")
cam.release()
cv2.destroyAllWindows()