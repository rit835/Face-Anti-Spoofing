from live.livenessnet import LivenessNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import np_utils
from imutils import paths
import numpy as np
import pickle
import cv2
import os
INIT_LR = 1e-4
BS = 8
EPOCHS = 70
imagePaths = list(paths.list_images("dataset"))
data = []
labels = []
for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (32, 32))
    data.append(image)
    labels.append(label)
data = np.array(data, dtype="float") / 255.0
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = np_utils.to_categorical(labels, 2)
(trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=0.25, random_state=42)
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,horizontal_flip=True, fill_mode="nearest")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model = LivenessNet.build(width=32, height=32, depth=3,classes=len(le.classes_))
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,epochs=EPOCHS)
model.save("live.model")
f = open("live.pickle", "wb")
f.write(pickle.dumps(le))
f.close()
