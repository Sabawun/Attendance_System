from datetime import time
import cv2
import kivy
import numpy as np
from kivy.clock import Clock
from kivy.lang import Builder
from kivymd.app import MDApp
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.uix.label import Label
from kivymd.uix.boxlayout import MDBoxLayout
from kivy.uix.screenmanager import ScreenManager
from kivy.uix.screenmanager import Screen
from kivy.uix.camera import Camera
from tensorflow import keras
from Source.CNN.Sabawun_PCA_Attendance import pca

User = "Sabawun"

CATEGORIES = ["Sabawun", "Other"]


model = keras.models.load_model("/Users/sabawunafzalkhattak/Desktop/Attendance_System/CNN_Models/" + User)

classifier = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

Window.size = (320, 600)


class Login(Screen):
    pass


class Home(Screen):
    pass


class Course(Screen):
    pass


class Recognition(Screen):
    pass


class KivyCamera(Image):

    def __init__(self, fps=30, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / fps)
        self.actual_fps = []

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = classifier.detectMultiScale(gray, 1.5, 5)

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return frame

    def frame_to_texture(self, frame):
        # convert it to texture
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tobytes()
        image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        return image_texture

    def update(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return

        if self.detect:
            self.detect_faces(frame)

        # display image from the texture
        self.texture = self.frame_to_texture(frame)


class Manager(ScreenManager):
    pass


class Attendance(MDApp):

    def build(self):
        self.theme_cls.theme_style = "Light"
        self.theme_cls.primary_palette = "Red"
        return Builder.load_file('Attendance.kv')


Attendance().run()
