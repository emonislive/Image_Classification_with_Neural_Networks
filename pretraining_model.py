import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from keras import datasets, layers, models

# ? training and testing data display format [DATASET CIFAR-10]
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
# ? preparing the data by assigning pixels to 0-1 out of 255 (scaling down the data)
training_images, testing_images = training_images / 255, testing_images / 255

# ! the order of the objects in the dataset [DATASET CIFAR-10]
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

training_images = training_images[:50000]
training_labels = training_labels[:50000]
testing_images = testing_images[:10000]
testing_labels = testing_labels[:10000]

# ! For Training
model = models.Sequential([
    layers.Input(shape=(32,32,3)),  # Define input layer explicitly
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=100, validation_data=(testing_images, testing_labels))
loss, accuracy = model.evaluate(testing_images, testing_labels)

print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

model.save('image_classifier.keras')

# ! After saving the trained model
model = models.load_model('image_classifier.keras')
