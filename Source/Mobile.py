from datetime import time

import cv2
import kivy
import numpy as np
from kivy.lang import Builder
from kivymd.app import MDApp
from kivy.uix.label import Label
from kivymd.uix.boxlayout import MDBoxLayout
from kivy.uix.screenmanager import ScreenManager
from kivy.uix.screenmanager import Screen
from kivy.uix.camera import Camera
from tensorflow import keras
from Source.CNN.Sabawun_PCA_Attendance import pca


CATEGORIES = ["Sabawun", "Other"]

User = "Sabawun"

model = keras.models.load_model("/Users/sabawunafzalkhattak/Desktop/Attendance_System/CNN_Models/" + User)


class Login(Screen):
    pass


class Home(Screen):
    pass


class Course(Screen):
    pass


classifier = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")


def recognize(cam):
    gray = cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, 1.5, 5)
    Image_test = []
    for face in faces:
        x, y, w, h = face
        im_face = cam[y:y + h, x:x + w]  # frame with only detected face
        new_array = cv2.resize(im_face, (64, 64))
        Face_image = np.array(new_array)
        Face_image = Face_image / 255.0
        Face_image = Face_image.reshape(-1, 64, 64, 3)
        Face_image_flat = Face_image.reshape(-1, 12288)
        Face_image_pca = pca.transform(Face_image_flat)
        Image_test.append(np.array(Face_image_pca))

    for i, face in enumerate(faces):
        predict_x = model.predict(np.array(Image_test[i]))
        classes_x = CATEGORIES[int(np.argmax(predict_x, axis=1))]
        print(classes_x)


class Recognition(Screen):
    def capture(self):
        cam = self.ids['camera']
        recognize(cam)


class Manager(ScreenManager):
    pass


class Attendance(MDApp):

    def build(self):
        self.theme_cls.theme_style = "Light"
        self.theme_cls.primary_palette = "Red"
        return Builder.load_file('Attendance.kv')


Attendance().run()
