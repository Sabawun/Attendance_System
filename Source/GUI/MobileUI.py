import sys
from cv2 import cv2
import numpy as np
from kivy.app import App
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.uix import popup
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.uix.widget import Widget
from kivymd.app import MDApp
from kivy.uix.image import Image
from kivy.properties import ObjectProperty
from kivy.core.window import Window
from kivy.uix.label import Label
from kivymd.uix.boxlayout import MDBoxLayout
from kivy.uix.screenmanager import ScreenManager, FadeTransition
from kivy.uix.screenmanager import Screen
from kivy.graphics.texture import Texture
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.textfield import MDTextField
from tensorflow import keras
from Source.CNN.PCA_Attendance import PCA_Attendance
from datetime import datetime
from datetime import timedelta
from datetime import date
import calendar


class WindowManager(ScreenManager):
    pass


User = ""
ID = ""
sm = WindowManager()
surname = ""
model = ""
p = ""
CATEGORIES = []

Window.size = (320, 600)


class PopupWindow(Widget):
    def btn(self):
        popFun("button")


class P(FloatLayout):
    pass


# function that displays the content
def passwordPop():
    show = P()
    password = ObjectProperty(None)
    box = BoxLayout(orientation='vertical', padding=10)
    box.add_widget(Label(text="Please enter new password", font_size=16, color=(.8, .8, .8, 1)))
    test = MDTextField(mode="rectangle", hint_text="password", error_color=(1, 1, 1, 0), icon_right="lock",
                       current_hint_text_color=(0.6, 0.6, 0.8, 1))
    box.add_widget(test)
    btn1 = MDRaisedButton(text="Save")
    box.add_widget(btn1)
    window = Popup(title="New password", content=box,
                   size_hint=(None, None), auto_dismiss=False, size=(320, 300))
    btn1.bind(on_press=window.dismiss)
    window.open()
    print(test.text)


def popFun(message):
    show = P()
    btn1 = MDRaisedButton(text="Close", md_bg_color=(.4, 0, .15, 1))
    window = Popup(title=message, content=btn1,
                   size_hint=(None, None), auto_dismiss=False, size=(320, 300))
    btn1.bind(on_press=window.dismiss)
    window.open()


def popFun5(index):
    show = P()
    window = Popup(title="Please try again after lecture starts.", content=show,
                   size_hint=(None, None), size=(320, 100))
    window.open()


class Login(Screen):
    studentId = ObjectProperty(None)
    password = ObjectProperty(None)

    def logger(self):

        global User
        # User = check_username_password(self.ids.studentId.text, self.ids.password.text)
        user = "2243459"
        User = "Onur"
        global surname
        surname = "C. Güven"
        global ID

        ID = user
        if self.ids.studentId.text != user:
            popFun("ID not found.")
        else:
            if self.ids.password.text == '1':
                passwordPop()
                self.manager.current = "Course"
                self.ids.studentId.text = ""
                self.ids.password.text = ""
            else:
                popFun("Welcome, %s" % user)
                self.manager.current = "Course"
                self.ids.studentId.text = ""
                self.ids.password.text = ""

    def exiter(self):
        sys.exit()


classifier = cv2.CascadeClassifier("../ImageProcessing/haarcascade_frontalface_alt.xml")

course = ""
pca, train_img_pca, test_img_pca = 0, 0, 0
count = 0


def attendancer(instance):
    global course
    course = instance.text
    App.get_running_app().root.transition = FadeTransition(duration=.3)
    App.get_running_app().root.current = "VideoScreen"


def frame_to_texture(frame):
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
                if self.totalcount > 10:  # success, take attendance
                    popFun("Attendance successful.")
                    App.get_running_app().root.transition = FadeTransition(duration=.3)
                    App.get_running_app().root.current = "Course"
                    self.detect = 0
                    self.capture.release()
                else:
                    popFun("Attendance failed.")
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
        global CATEGORIES, p, model, pca, train_img_pca, test_img_pca
        CATEGORIES = [User, "Other"]
        p = "../../CNN_Models/" + str(User)
        model = keras.models.load_model(p)
        pca, train_img_pca, test_img_pca = PCA_Attendance(User)
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
    def build(self):
        widget = FloatLayout(size=(300, 300))
        nameLabel = Label(text=("[b]%s %s, %s[/b]" % (User, surname, ID)), marker=True, font_size=11,
                          color=(0, 0, 0, 1),
                          pos=(-60, 280))
        widget.add_widget(nameLabel)
        return self.add_widget(nameLabel)

    nam = ""

    box_situation = 0
    my_label = 0
    now = datetime.now()
    course_situation = 0
    clock_situation = 0

    def returntoLogin(self):
        self.box_situation = 0
        self.course_situation = 0
        self.removeWid()
        self.manager.current = "Login"

    def dropDownMenu(self):

        if self.box_situation == 1:
            pass
        else:
            self.course_situation = 1
            LabelX, LabelY = -62, 230
            positionY = 511
            if self.clock_situation == 0:
                Clock.schedule_interval(self.update_clock, 1)
                self.clock_situation = 1
            curr_date = date.today()
            self.my_label = Label(text="                          %s %s" % (self.now.strftime('%H:%M:%S'),
                                                                            calendar.day_name[curr_date.weekday()]),
                                  pos=(LabelX, 260),
                                  font_size=15, color=(0, 0, 0, 1))
            # The label is the only widget in the interface
            LectureList = ["CNG445", "CNG492", "CNG482", "CNG457"]
            LectureSchedules = ["Monday,8:40,10:30,Wednesday,13:40,14:30", "Wednesday,14:40,16:30",
                                "Friday,10:40,13:30", "Tuesday,11:40,12:30,Sunday,16:44,23:30"]  # get this with query.

            Today = calendar.day_name[curr_date.weekday()]
            now = datetime.now()

            todayHour = datetime.strptime(now.strftime('%H:%M'), "%H:%M")

            my_box = FloatLayout(size=(320, 600))
            my_box.my_buttons = []
            labelDivider = Label(text="-----------------------------------------------------------------------------"
                                      "----------------------------------------------------------------------------",
                                 pos=(LabelX, LabelY + 20),
                                 font_size=11, color=(0, 0, 0, 1))
            my_box.my_buttons.append(labelDivider)
            my_box.add_widget(labelDivider)
            my_box.my_buttons.append(self.my_label)
            my_box.add_widget(self.my_label)
            for i in range(len(LectureSchedules)):

                Day, start, end, *args = LectureSchedules[i].split(',')
                startHour = datetime.strptime(start, "%H:%M")
                endHour = datetime.strptime(end, "%H:%M")
                labelDivider = Label(
                    text="-----------------------------------------------------------------------------"
                         "----------------------------------------------------------------------------",
                    pos=(LabelX, LabelY - 20),
                    font_size=11, color=(0, 0, 0, 1))
                my_box.my_buttons.append(labelDivider)
                my_box.add_widget(labelDivider)
                labelLecTime = Label(text="%s --> %s %s %s"
                                          % (LectureList[i], Day, start, end), pos=(LabelX, LabelY),
                                     font_size=12, color=(0, 0, 0, 1))

                LabelY -= 40
                if Today == Day:  # check hours if day is same
                    if startHour < todayHour < endHour:
                        button = MDRaisedButton(text="Attend", font_size=12, elevation=20,
                                                pos=(210, positionY), md_bg_color=(0, .5, 0, 1))
                        positionY -= 40
                        button.bind(on_press=attendancer)
                        my_box.my_buttons.append(labelLecTime)
                        my_box.my_buttons.append(button)
                        my_box.add_widget(labelLecTime)
                        my_box.add_widget(labelDivider)
                        my_box.add_widget(button)
                    else:
                        button = MDRaisedButton(text="Attend", font_size=12, elevation=20,
                                                pos=(210, positionY), md_bg_color=(.6, 0, 0, 1))
                        positionY -= 40
                        button.bind(on_press=popFun5)
                        my_box.my_buttons.append(labelLecTime)
                        my_box.my_buttons.append(button)
                        my_box.add_widget(labelLecTime)
                        my_box.add_widget(button)
                else:
                    button = MDRaisedButton(text="Attend", font_size=12, elevation=20,
                                            pos=(210, positionY), md_bg_color=(.6, .0, .0, 1))
                    positionY -= 40
                    button.bind(on_press=popFun5)
                    my_box.my_buttons.append(labelLecTime)
                    my_box.my_buttons.append(button)
                    my_box.add_widget(labelLecTime)
                    my_box.add_widget(button)
                argsS = ""
                if args:  # check the next schedule of the current lecture
                    labelDivider = Label(
                        text="-----------------------------------------------------------------------------"
                             "----------------------------------------------------------------------------",
                        pos=(LabelX, LabelY - 20),
                        font_size=11, color=(0, 0, 0, 1))
                    my_box.my_buttons.append(labelDivider)
                    my_box.add_widget(labelDivider)
                    for y in range(len(args)):  # convert string to format then split
                        argsS = argsS + args[y] + ","
                    Day, start, end, *args = argsS.split(',')
                    labelLecTime2 = Label(text="%s --> %s %s %s" % (LectureList[i], Day, start, end),
                                          pos=(LabelX, LabelY),
                                          font_size=11, color=(0, 0, 0, 1))
                    LabelY -= 40
                    startHour = datetime.strptime(start, "%H:%M")
                    endHour = datetime.strptime(end, "%H:%M")
                    if Today == Day:  # check hours if day is same
                        if startHour < todayHour < endHour:
                            button = MDRaisedButton(text="Attend", font_size=12, elevation=20,
                                                    pos=(210, positionY), md_bg_color=(.0, .5, .0, 1))
                            positionY -= 40
                            button.bind(on_press=attendancer)
                            my_box.my_buttons.append(labelLecTime2)
                            my_box.my_buttons.append(button)
                            my_box.add_widget(labelLecTime2)
                            my_box.add_widget(button)
                        else:
                            button = MDRaisedButton(text="Attend", font_size=12, elevation=20,
                                                    pos=(210, positionY), md_bg_color=(.6, .0, .0, 1))
                            positionY -= 40
                            button.bind(on_press=popFun5)
                            my_box.my_buttons.append(labelLecTime2)
                            my_box.my_buttons.append(button)
                            my_box.add_widget(labelLecTime2)
                            my_box.add_widget(button)
                    else:
                        button = MDRaisedButton(text="Attend", font_size=12, elevation=20,
                                                pos=(210, positionY), md_bg_color=(.6, .0, .0, 1))
                        positionY -= 40
                        button.bind(on_press=popFun5)
                        my_box.my_buttons.append(labelLecTime2)
                        my_box.my_buttons.append(button)
                        my_box.add_widget(labelLecTime2)
                        my_box.add_widget(button)
            self.box_situation = 1
            return self.add_widget(my_box)

    def removeWid(self):
        if self.box_situation == 1:
            self.box_situation = 0
            self.clear_widgets(self.children[:1])

    def update_clock(self, *args):

        curr_date = date.today()
        self.now = self.now + timedelta(seconds=1)
        self.my_label.text = "                          %s, %s" % (self.now.strftime('%H:%M:%S'),
                                                                   calendar.day_name[curr_date.weekday()])

    def refresh(self):
        print(self.course_situation)
        if self.course_situation == 1:
            self.removeWid()
            self.dropDownMenu()


sm.add_widget(Login(name="Login"))
sm.add_widget(Course(name="Course"))
sm.add_widget(VideoScreen(name="VideoScreen"))


class loginMain(MDApp):
    def build(self):
        kv = Builder.load_file('Login.kv')
        return kv


loginMain().run()
