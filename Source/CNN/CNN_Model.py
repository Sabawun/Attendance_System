import numpy as np
from keras.utils import np_utils
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from PCA_Attendance import PCA_Attendance


def CNN_Model(User):
    pca, train_img_pca, test_img_pca = PCA_Attendance(User)
    Training_Labels = np.load("../Files/" + User +
                              "/" + User + "_Train_Labels.npy")
    Testing_Labels = np.load("../Files/" + User +
                             "/" + User + "_Test_Labels.npy")

    y_train = np_utils.to_categorical(Training_Labels)
    y_test = np_utils.to_categorical(Testing_Labels)
    batch_size = 10
    num_classes = 2
    epochs = 35

    model = Sequential()
    model.add(Dense(512, activation='sigmoid', input_shape=(pca.n_components_,)))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    # model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(16, activation='relu'))
    # model.add(Dropout(0.4))
    # model.add(Dense(256, activation='relu'))
    # model.add(Dense(256, activation='relu'))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.4))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dropout(0.4))
    # model.add(Dense(16, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])

    history = model.fit(train_img_pca, y_train, batch_size=batch_size, epochs=epochs,
                        validation_data=(test_img_pca, y_test))

    p = Path(__file__).parents[2]
    p = '{0}\\CNN_Models\\{1}'.format(str(p), User)
    model.save(p)
