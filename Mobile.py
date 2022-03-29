from kivy.lang import Builder
from kivymd.app import MDApp
from kivy.uix.screenmanager import ScreenManager, Screen
import cv2
import numpy as np
from tkinter import messagebox
from tensorflow import keras
from PCA_Attendance import pca

CATEGORIES = np.load("Categories.npy")

model = keras.models.load_model("Face_Recognizer")


count = 0
dic = dict()
result = []
first_click = True
last = ""


# Screens

class Login(Screen):
    pass


class Home(Screen):
    pass


class Attendance(Screen):
    pass


class Window(ScreenManager):
    pass


class MainApp(MDApp):

    def build(self):
        self.theme_cls.theme_style = "Light"
        self.theme_cls.primary_palette = "Red"
        sm = Builder.load_file("Screens.kv")
        sm.add_widget(Login(name='Login'))
        sm.add_widget(Home(name='Home'))
        sm.add_widget(Attendance(name='Attendance'))
        return sm

    def quit(self):
        exit()

    def take_attendance(self):
        cap = cv2.VideoCapture(0)
        classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        while True:

            ret, frame = cap.read()

            # turn the frame to gray for easier detection & recognition.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detects the faces in a frame
            faces = classifier.detectMultiScale(gray, 1.5, 5)

            # array for storing all processed faces in the frame
            Image_test = []
            # checks all the faces in the frame, and stores them on X.test numpy array.
            for face in faces:
                x, y, w, h = face
                im_face = frame[y:y + h, x:x + w]  # frame with only detected face
                new_array = cv2.resize(im_face, (64, 64))
                Face_image = np.array(new_array)
                Face_image = Face_image / 255.0
                Face_image = Face_image.reshape(-1, 64, 64, 3)
                Face_image_flat = Face_image.reshape(-1, 12288)
                Face_image_pca = pca.transform(Face_image_flat)
                Image_test.append(np.array(Face_image_pca))

            # if any faces in the frame, then call predict method of knn to calculate the distances.
            if len(faces) > 0:
                # draws rectangles on a frame,.
                for i, face in enumerate(faces):
                    predict_x = model.predict(np.array(Image_test[i]))  # or this
                    classes_x = CATEGORIES[int(np.argmax(predict_x, axis=1))]
                    x, y, w, h = face
                    # print("i = " + str(i))
                    # print(classes_x)
                    # print(predict_x)
                    test = predict_x[0]
                    i = 0
                    for categories in CATEGORIES:
                        if classes_x == categories:
                            break
                        else:
                            i = i + 1
                    predict = predict_x[0]
                    # print(predict[i])
                    if predict[i] > 0.9998:
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
                    # /////
                    global dic
                    global count
                    #

                    count += 1

                    if dic.get(str(classes_x)):
                        dic[str(classes_x)] += 1
                    else:
                        dic[str(classes_x)] = 1
                    # print(count, " ", str(classes_x))

            if count > 2000:
                messagebox.showinfo(title='Done', message='Successfully recorded students.')
                cap.release()
                cv2.destroyAllWindows()
                break
            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                break
            cv2.imshow("feed", frame)


if __name__ == '__main__':
    MainApp().run()
