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
from Source.CNN.PCA_Attendance import PCA_Attendance
from pathlib import Path
from kivy.properties import ObjectProperty

User = "Sabawun"
pca, train_img_pca, test_img_pca = PCA_Attendance(User)


CATEGORIES = [User, "Other"]

p = "../../CNN_Models/" + User
model = keras.models.load_model(p)

classifier = cv2.CascadeClassifier("../ImageProcessing/haarcascade_frontalface_alt.xml")

Window.size = (320, 600)


class Login(Screen):
    studentId = ObjectProperty(None)
    password = ObjectProperty(None)
    user = ""

    def logger(self):
        # User = check_username_password(self.ids.studentId.text, self.ids.password.text)
        user = "Sabawun"
        print(self.ids.studentId.text)
        print(self.ids.password.text)

        if user == "not":
            self.ids.studentId.text = " "
            self.ids.password.text = ""
        else:
            self.parent.current = "Recognition"


class Course(Screen):
    pass


class Recognition(Screen):
    pass


def frame_to_texture(frame):
    # convert it to texture
    buf1 = cv2.flip(frame, 0)
    buf = buf1.tobytes()
    image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
    image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
    return image_texture


def detect_faces(frame):
    print(Login.user)
    faces = classifier.detectMultiScale(frame, 1.5, 5)
    Image_test = []

    # Draw a rectangle around the faces
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

    if len(faces) > 0:
        for i, face in enumerate(faces):
            predict_x = model.predict(np.array(Image_test[i]))
            classes_x = CATEGORIES[int(np.argmax(predict_x, axis=1))]
            x, y, w, h = face
            i = 0
            for categories in CATEGORIES:
                if classes_x == categories:
                    break
                else:
                    i = i + 1
            predict = predict_x[0]
            if predict[i] > 0.98:
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

    return frame


class KivyCamera(Image):

    def __init__(self, fps=30, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / fps)
        self.actual_fps = []

    def update(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return

        if self.detect:
            detect_faces(frame)

        # display image from the texture
        self.texture = frame_to_texture(frame)


class Manager(ScreenManager):
    pass


class Attendance(MDApp):

    def build(self):
        self.theme_cls.theme_style = "Light"
        self.theme_cls.primary_palette = "Red"
        mapp = Builder.load_file('Attendance.kv')
        return mapp


Attendance().run()
