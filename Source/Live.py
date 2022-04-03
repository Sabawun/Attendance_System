from cv2 import cv2
import numpy as np
import keras
from PCA_Attendance import pca, train_img_pca, test_img_pca
# from Training_Testing import CATEGORIES
# CATEGORIES = ["Alpsen", "Onur", "Sabawun"]
from CNN import model

CATEGORIES = np.load("/Users/sabawunafzalkhattak/Desktop/Attendance_System/Source/DataFiles/Categories.npy")

model = keras.models.load_model("/Users/sabawunafzalkhattak/Desktop/Attendance_System/Source/Face_Recognizer")


def begin():
    cap = cv2.VideoCapture(0)

    # viola jones algorithm for feature extraction,finds the face object.
    classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # looping for constant video stream and detection & recognition.
    while True:

        ret, frame = cap.read()
        faces = classifier.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, )
        Image_test = []
        for face in faces:
            x, y, w, h = face
            im_face = frame[y:y + h, x:x + w]  # frame with only detected face
            new_array = cv2.resize(im_face, (64, 64))
            # img = cv2.flip(img_array, 1)
            Face_image = np.array(new_array)
            Face_image = Face_image / 255.0
            Face_image = Face_image.reshape(-1, 64, 64, 3)
            Face_image_flat = Face_image.reshape(-1, 12288)
            Face_image_pca = pca.transform(Face_image_flat)
            Image_test.append(np.array(Face_image_pca))
        if len(faces) > 0:
            # draws rectangles on a frame,.
            for i, face in enumerate(faces):
                predict_x = model.predict(np.array(Image_test[i]))  # or this
                classes_x = CATEGORIES[int(np.argmax(predict_x, axis=1))]
                x, y, w, h = face
                # print("i = " + str(i))
                print(classes_x)
                print(predict_x)
                test = predict_x[0]
                i = 0
                for categories in CATEGORIES:
                    if classes_x == categories:
                        break
                    else:
                        i = i + 1
                predict = predict_x[0]
                print(predict[i])
                if predict[i] > 0.9:
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (0, 250, 0), 3)
                    cv2.putText(frame, str(classes_x), (x - 50, y - 10),
                                cv2.FONT_HERSHEY_PLAIN, 2,
                                (0, 250, 0), 2)
                    cv2.putText(frame, str(predict[i]), (x - 50, y - 50),
                                cv2.FONT_HERSHEY_PLAIN, 2,
                                (250, 0, 0), 2)
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (0, 0, 250), 3)
        cv2.imshow("Video Feed", frame)

        key = cv2.waitKey(50)
        # Press q to end stream and recognition.
        if key & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


begin()
