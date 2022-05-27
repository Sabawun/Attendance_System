import sys
from datetime import time
import cv2
import kivy
import numpy as np
from kivy.app import App
from kivy.base import runTouchApp
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.popup import Popup
from kivy.uix.widget import Widget
from kivymd.app import MDApp
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.uix.label import Label
from kivymd.uix.boxlayout import MDBoxLayout
from kivy.uix.screenmanager import ScreenManager, FadeTransition
from kivy.uix.screenmanager import Screen
from kivy.uix.camera import Camera
# from tensorflow import keras
# from Source.CNN.PCA_Attendance import PCA_Attendance
from pathlib import Path
from kivy.properties import ObjectProperty
from kivymd.uix.button import MDFillRoundFlatButton, MDRoundFlatButton, MDRaisedButton
from tensorflow import keras
from Source.CNN.PCA_Attendance import PCA_Attendance


class WindowManager(ScreenManager):
    pass


# kv file
User = ""
sm = WindowManager()
# adding screens

Window.size = (320, 600)


class PopupWindow(Widget):
    def btn(self):
        popFun()


class P(FloatLayout):
    pass


# function that displays the content
def popFun():
    show = P()
    window = Popup(title="ID not recognized.\nPlease try again.", content=show,
                   size_hint=(None, None), size=(320, 100))
    window.open()


def popFun2():
    show = P()
    window = Popup(title="Login Success", content=show,
                   size_hint=(None, None), size=(120, 100))
    window.open()

def popFun3():
    show = P()
    window = Popup(title="User not recognized,Please try again.", content=show,
                   size_hint=(None, None), size=(320, 100))
    window.open()

def popFun4():
    show = P()
    window = Popup(title="Attendance registered successfully.", content=show,
                   size_hint=(None, None), size=(320, 100))
    window.open()

def popFun5(index):
    show = P()
    window = Popup(title="Please try again after lecture starts.", content=show,
                   size_hint=(None, None), size=(320, 100))
    window.open()

class Login(Screen):
    studentId = ObjectProperty(None)
    password = ObjectProperty(None)
    def pusher(self):
        my_box = MDBoxLayout(orientation="vertical")
        my_box.wids = []
        my_box.wids.append(my_box)
        my_box.add_widget(my_box)
        return self.add_widget(my_box)
    def logger(self):

        global User
        # User = check_username_password(self.ids.studentId.text, self.ids.password.text)
        user = "2243459"
        User = "Onur"
        if self.ids.studentId.text != user:
            popFun()
        else:
            popFun2()
            self.manager.current = "Course"
            self.ids.studentId.text = " "
            self.ids.password.text = ""

    def exiter(self):
        sys.exit()


classifier = cv2.CascadeClassifier("../ImageProcessing/haarcascade_frontalface_alt.xml")
User = "Onur"
CATEGORIES = [User, "Other"]
p = "../../CNN_Models/" + str(User)
model = keras.models.load_model(p)
course = ""
pca, train_img_pca, test_img_pca = PCA_Attendance(User)
count = 0




def attendancer(instance):
    global course
    course = instance.text
    App.get_running_app().root.transition = FadeTransition(duration=.3)
    App.get_running_app().root.current = "VideoScreen"


def frame_to_texture(frame):
    # convert it to texture
    buf1 = cv2.flip(frame, 0)
    buf = buf1.tobytes()
    image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
    image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
    return image_texture


def detect_faces(frame):
    faces = classifier.detectMultiScale(frame, 1.5, 5)
    Image_test = []
    global count
    count = 0
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
            if predict[i] > 0.51:
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
            print(classes_x, CATEGORIES[0])

            if classes_x == CATEGORIES[0]:
                count += 1
            else:
                print("Not Equ")
                print(type(classes_x), type(CATEGORIES))
                count = 0

    return frame, count


class KivyCamera(Image):
    def __init__(self, fps=30, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 30)
        self.actual_fps = []
        self.detect = 1
        self.totalframes = 0
        self.totalcount = 0
        self.count = 0

    def update(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return

        if self.detect:
            frame, self.count = detect_faces(frame)
            self.totalframes += 1
            self.totalcount = self.totalcount + self.count
            print(self.totalcount)
            if self.totalframes > 100:
                if self.totalcount > 20:    # success, take attendance
                    popFun4()
                    App.get_running_app().root.transition = FadeTransition(duration=.3)
                    App.get_running_app().root.current = "Course"
                    self.detect = 0
                    self.capture.release()
                else:
                    popFun3()
                    App.get_running_app().root.transition = FadeTransition(duration=.3)
                    App.get_running_app().root.current = "Course"
                    self.capture.release()
                    self.totalcount = 0
                    self.totalframes = 0
                    self.count = 0
        else:
            self.capture.release()

        # display image from the texture
        self.texture = frame_to_texture(frame)


class VideoScreen(Screen):
    kivyc = None

    def cameraMethod(self):
        my_box = MDBoxLayout(orientation="vertical")
        my_box.wids = []
        self.kivyc = KivyCamera()
        my_box.wids.append(self.kivyc)
        my_box.add_widget(self.kivyc)
        return self.add_widget(my_box)

    def removeCamera(self):
        self.kivyc.detect = 0
        self.clear_widgets(self.children[:1])
        del self.kivyc


class Course(Screen):
    nam = ""
    box_situation = 0
    def returntoLogin(self):
        self.manager.current = "Login"

    def dropDownMenu(self):
        red = [.75, 0, 0, 1]
        green = [0, .75, 0, 1]
        blue = [0, .4, .4, 1]
        purple = [1, 0, 1, 1]
        LectureList = ["CNG445", "CNG492", "CNG482", "CNG457"]
        LectureSchedules = ["Monday 8:40-10:30" + " Wednesday 13:40-14:30", "Wednesday 14:40-16:30",
                            "Friday 10:40-13:30", "Tuesday 11:40-12:30" + " Thursday 10:40-12:30"]
        my_box = MDBoxLayout(orientation="vertical")
        my_box2 = MDBoxLayout(orientation="vertical")
        my_box3 = MDBoxLayout(orientation="vertical")
        my_box4 = MDBoxLayout(orientation="vertical")
        my_box.my_buttons = []
        labelCurrentTime = Label(text="Current Date & Time : 15:37 Wednesday",halign="center", valign="middle",
                                 color=(0, 0, 0, 1))
        my_box.my_buttons.append(my_box2)
        my_box.add_widget(my_box2)
        my_box.my_buttons.append(my_box4)
        my_box.add_widget(my_box4)
        my_box.my_buttons.append(labelCurrentTime)
        my_box.add_widget(labelCurrentTime)
        x=0
        for i in range(len(LectureList)):
            labelLecTime = Label(text=LectureList[i] + "\n" + LectureSchedules[i], halign="center", valign="middle",
                                 font_size=11, color=(0, 0, 0, 1))
            if x<3:
                button = MDFillRoundFlatButton(text="Attend", font_size=11, elevation=20,pos=(100,100),
                                               pos_hint={"center_x": 0.5, "center_y": 0.4}, md_bg_color=red)
                button.bind(on_press=popFun5)
                my_box.my_buttons.append(labelLecTime)
                my_box.my_buttons.append(button)
                my_box.add_widget(labelLecTime)
                my_box.add_widget(button)
                x +=1
            else:
                button = MDFillRoundFlatButton(text="Attend", font_size=12, elevation=20,
                                               pos_hint={"center_x": 0.5, "center_y": 0.4}, md_bg_color=green)
                button.bind(on_press=attendancer)
                my_box.my_buttons.append(labelLecTime)
                my_box.my_buttons.append(button)
                my_box.add_widget(labelLecTime)
                my_box.add_widget(button)
                x+=1
        self.box_situation = 1
        my_box.my_buttons.append(my_box3)
        my_box.add_widget(my_box3)
        return self.add_widget(my_box)

    def removeWid(self):
        if self.box_situation == 1:
            self.box_situation = 0
            self.clear_widgets(self.children[:1])


sm.add_widget(Login(name="Login"))
sm.add_widget(Course(name="Course"))
sm.add_widget(VideoScreen(name="VideoScreen"))


# class that builds gui
class loginMain(MDApp):
    def build(self):
        kv = Builder.load_file('Login.kv')
        return kv


loginMain().run()
