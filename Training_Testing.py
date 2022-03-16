import random
import numpy as np
import os
import cv2
from tqdm import tqdm

DATADIR = "/Users/sabawunafzalkhattak/Desktop/Attendance_System/Dataset/Training_Set"
DATADIR2 = "/Users/sabawunafzalkhattak/Desktop/Attendance_System/Dataset/Testing_Set"
# DATADIR0 = "E:\Attendance_System"

# CATEGORIES = ["Abdullah", "Alpsen", "Areesh", "Khizr", "Mohsin", "Nazal" ,"Omar", "Onur", "Osama", "Ramis", "Sabawun"]
CATEGORIES = ["Abdullah", "Alpsen", "Sabawun", "Onur"]


#######################################################################################################################
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)  # create path
        class_num = CATEGORIES.index(category)  # get the classification

        for img in tqdm(os.listdir(path)):  # iterate over each image
            try:
                img_array = cv2.imread(os.path.join(path, img))  # convert to array
                new_array = cv2.resize(img_array, (64, 64))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:
                pass


training_data = []
create_training_data()

random.shuffle(training_data)
Training_Images = []
Training_Labels = []

for features, label in training_data:
    Training_Images.append(features)
    Training_Labels.append(label)

Training_Images = np.array(Training_Images).reshape(-1, 64, 64, 3)

Norm_Training_Images = Training_Images / 255.0
Norm_Training_Images = np.array(Norm_Training_Images)
Training_Labels = np.array(Training_Labels)

Flatten_Norm_Training_Images = Norm_Training_Images.reshape(-1, 12288)


#######################################################################################################################

def create_testing_data():
    for category in CATEGORIES:  #

        path = os.path.join(DATADIR2, category)  # create path
        class_num = CATEGORIES.index(category)  # get the classification

        for img in tqdm(os.listdir(path)):  # iterate over each image per faces
            try:
                img_array = cv2.imread(os.path.join(path, img))  # convert to array
                new_array = cv2.resize(img_array, (64, 64))  # resize to normalize data size
                testing_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:
                pass


testing_data = []
create_testing_data()

random.shuffle(testing_data)

Testing_Images = []
Testing_Labels = []

for feature, labels in testing_data:
    Testing_Images.append(feature)
    Testing_Labels.append(labels)

Testing_Images = np.array(Testing_Images).reshape(-1, 64, 64, 3)

Norm_Testing_Images = Testing_Images / 255.0
Norm_Testing_Images = np.array(Norm_Testing_Images)
Testing_Labels = np.array(Testing_Labels)

Flatten_Norm_Testing_Images = Norm_Testing_Images.reshape(-1, 12288)

########################################################################################################################
Flatten_Norm_Testing_Images = np.save("Flatten_Norm_Testing_Images", Flatten_Norm_Testing_Images)
Flatten_Norm_Training_Images = np.save("Flatten_Norm_Training_Images", Flatten_Norm_Training_Images)
Training_Labels = np.save("Training_Labels", Training_Labels)
Testing_Labels = np.save("Testing_Labels", Testing_Labels)
