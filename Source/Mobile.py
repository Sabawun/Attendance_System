import kivy
from kivy.lang import Builder
from kivymd.app import MDApp
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout


class Attendance(MDApp):

    def build(self):
        self.theme_cls.theme_style = "Light"
        self.theme_cls.primary_palette = "Red"
        return Builder.load_file('Attendance.kv')

Attendance().run()
