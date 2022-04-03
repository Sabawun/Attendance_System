#This module is used to detect faces in images, crop them from pictures, and save for later training purposes.
import cv2
cam = cv2.VideoCapture(0)
cam.set(3, 1920) # 640
cam.set(4, 1080) # 480
#We are using Haar Cascades to detect faces from images.
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
print("\n Initializing...")
userName = input("What is the user's name?\n")
count = 0
while(True):
    #We receive video stream from the camera here.
    ret, img = cam.read()
    #We invert the camera vertically as it is inverted at the beggining defaultly.
    img = cv2.flip(img, 1)
    #We configure the method with our parameters. These parameters define the working style of the method and will impact the detection quality greatly.  
    faces = face_detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
    for (x,y,w,h) in faces:
        count += 1
        try:
            #After detecting any face, we try to crop the whole face with all the features in the scene.
            crop_img = img[y: y+int(h*1.2), x: x+w]
            cv2.imwrite(userName + "/Type("+ str(count) + ").jpg", crop_img)
        except:
            #If the boundaries we try to reach are out of reach, we simple crop what we have there.
            crop_img = img[y: y+h, x: x+w]
            cv2.imwrite(username + "/Type("+ str(count) + ").jpg", crop_img)
        cv2.imshow('image', crop_img)
    k = cv2.waitKey(100) & 0xff
    #After having 50 images or having pressed "Esc" button, we shut down the process.
    if k == 27:
        break
    elif count >= 50:
         break
print("\n Exiting Program")
cam.release()
cv2.destroyAllWindows()
