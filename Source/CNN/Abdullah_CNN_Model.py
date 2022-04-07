import numpy as np
from keras.utils import np_utils
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

from Abdullah_PCA_Attendance import pca, train_img_pca, test_img_pca

User = "Abdullah"

Training_Labels = np.load("/Users/sabawunafzalkhattak/Desktop/Attendance_System/Files/" + User +
                          "/" + User + "_Train_Labels.npy")
Testing_Labels = np.load("/Users/sabawunafzalkhattak/Desktop/Attendance_System/Files/" + User +
                         "/" + User + "_Test_Labels.npy")

y_train = np_utils.to_categorical(Training_Labels)
y_test = np_utils.to_categorical(Testing_Labels)

batch_size = 12  # 186 photos
num_classes = 2  # change to 4
epochs = 150

model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(pca.n_components_,)))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(16, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

history = model.fit(train_img_pca, y_train, batch_size=batch_size, epochs=epochs,
                    validation_data=(test_img_pca, y_test))

model.save("/Users/sabawunafzalkhattak/Desktop/Attendance_System/CNN_Models/" + User)
