from cv2 import cv2
import os
from tqdm import tqdm
DATADIR = "E:\Attendance_System\Alpsen"
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
CATEGORIES = ["Alpsen"]
print("\n Initializing...")
count = 0

cap = cv2.VideoCapture(0)

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)  # create path
    for img in tqdm(os.listdir(path)):  # iterate over each image
        try:
            print("t")
            img_array = cv2.imread(os.path.join(path, img))  # convert to array
            img = cv2.flip(img_array, 1)
            faces = face_detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
            for (x,y,w,h) in faces:
                print("face detect")
                count += 1
                try:
                    crop_img = img[y: y+int(h*1.2), x: x+w]
                    cv2.imwrite("E:\Attendance_System\Alpsen\Type("+ str(count) + ").jpg", crop_img)
                except:
                    crop_img = img[y: y+h, x: x+w]
                    cv2.imwrite("E:\Attendance_System\Alpsen\Type("+ str(count) + ").jpg", crop_img)
        except Exception as e:
            pass

print("\n Exiting Program")
