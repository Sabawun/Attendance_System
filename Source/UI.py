import csv
import datetime
from tkinter import *
from tkinter import messagebox
from tkinter import ttk
import cv2
import numpy as np
from tensorflow import keras

from Source.DatabaseConnection import check_username_password

User = check_username_password()

if User == "Abdullah":
    from CNN.Abdullah_PCA_Attendance import pca
elif User == "Sabawun":
    from CNN.Sabawun_PCA_Attendance import pca
elif User == "Alpsen":
    from CNN.Alpsen_PCA_Attendance import pca
elif User == "Onur":
    from CNN.Onur_PCA_Attendance import pca
else:
    print("User not specified")

CATEGORIES = [User, "Other"]
# from CNN import model
model = keras.models.load_model("/Users/sabawunafzalkhattak/Desktop/Attendance_System/CNN_Models/" + User)

count = 0
dic = dict()
result = []
first_click = True
last = ""
gui = Tk(className=' Attendance System')


def on_entry_click(event):
    # function that gets called whenever entry1 is clicked
    global first_click

    if first_click:  # if this is the first time they clicked it
        first_click = False
        lectureName.delete(0, "end")  # delete all the text in the entry


def recognizer():
    # load the faces

    # starts video stream from source 0, which is laptop camera.
    cap = cv2.VideoCapture(0)

    # viola jones algorithm for feature extraction,finds the face object.
    classifier = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

    # looping for constant video stream and detection & recognition.
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
                predict_x = model.predict(np.array(Image_test[i]))
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

        if count > 500:
            messagebox.showinfo(title='Done', message='Successfully recorded students.')
            cap.release()
            cv2.destroyAllWindows()
            break
        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            break
        cv2.imshow("feed", frame)


def reporter():
    global dic
    global result
    for i in dic.keys():
        # If we have captured the person for at least 100 times, we accept that he/she is in the class
        # We did this in order to discard false positives.
        if dic[i] > 100:
            result.append(i)
    messagebox.showinfo(title='Done', message='Record created successfully.')
    print("Found People:" + str(result))
    current_time = datetime.datetime.now()
    timeString = 'D_' + str(current_time.day) + '.' + str(current_time.month) + '.' + str(current_time.year) + '-T_' \
                 + str(current_time.hour) + '.' + str(current_time.minute) + '.' + str(current_time.second)
    lecture = lectureName.get()
    fileName = lecture + '-' + timeString + '.csv'
    with open(fileName, 'w') as f:

        # using csv.writer method from CSV package
        write = csv.writer(f)
        for i in result:
            write.writerow([i])


Label(gui, text="Attendance System", background='#3F374C', font='Aerial 15 bold italic').pack(pady=30)
gui.geometry("500x150")
gui['bg'] = '#3F374C'
recogButton = Button(gui, background='#787381', text='Start Taking Attendance', bd='5', command=recognizer)
reportButton = Button(gui, background='#787381', text='Create the Report', bd='5', command=reporter)
exitButton = Button(gui, background='#787381', text='Exit', bd='5', command=exit)

lectureName = ttk.Entry(gui)
lectureName.insert(0, 'Enter Course Code and Section:')
lectureName.bind('<FocusIn>', on_entry_click)

lectureName.pack(side='top', expand=True, fill=BOTH)
recogButton.pack(side='left', expand=True, fill=BOTH)
reportButton.pack(side='left', expand=True, fill=BOTH)
exitButton.pack(side='right', expand=True, fill=BOTH)

gui.mainloop()
