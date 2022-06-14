from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

Learning_rate = 1e-4
Epochs = 20
Batch_size = 32

directory = r"D:\Dataset"
categories = ["scabies", "sehat"]

data = []
labels = []

for category in categories:
    path = os.path.join(directory, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)

data.append(image)
labels.append(category)

LB = LabelBinarizer()
labels = LB.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

augmentation = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))


head_model = base_model.output
head_model = AveragePooling2D(pool_size=(7,7))(head_model)
head_model = Flatten(name="flatten")(head_model)
head_model = Dense(128, activation="relu")(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(2, activation="softmax")(head_model)

model = Model(inputs=base_model.input, outputs=head_model)

for layer in base_model.layers:
    layer.trainable = False

optimal = Adam(Lr=Learning_rate, decay=Learning_rate / Epochs)
model.compile(loss="binary_crossentropy", optimizer=optimal, metrics=["accuracy"])

training = model.fit(
    augmentation.flow(trainX, trainY, batch_size=Batch_size),
    steps_per_epoch=len(trainX) // Batch_size,
    validation_data=(testX, testY),
    validation_steps=len(testX) // Batch_size,
    epochs=Epochs
)

predIct = model.predict(testX, batch_size=Batch_size)

predIct = np.argmax(predIct, axis=1)

print(classification_report(testY.argmax(axis=1), predIct, target_names=LB.classes_))

model.save("skin_disease_detector.model", save_format="h5")

show = Epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, show), training.history["loss"], label="training_loss")
plt.plot(np.arange(0, show), training.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, show), training.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, show), training.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")