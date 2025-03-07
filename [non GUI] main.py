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

# Testing Dataset Preview
# for i in range(16):
#     plt.subplot(4,4, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(training_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[training_labels[i][0]])
#
# plt.show()

training_images = training_images[:40000]
training_labels = training_labels[:40000]
testing_images = testing_images[:6000]
testing_labels = testing_labels[:6000]

# For Training
# model = models.Sequential([
#     layers.Input(shape=(32,32,3)),  # Define input layer explicitly
#     layers.Conv2D(32, (3,3), activation='relu'),
#     layers.MaxPooling2D(2,2),
#     layers.Conv2D(64, (3,3), activation='relu'),
#     layers.MaxPooling2D(2,2),
#     layers.Conv2D(64, (3,3), activation='relu'),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(10, activation='softmax')
# ])
#
#
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(training_images, training_labels, epochs=20, validation_data=(testing_images, testing_labels))
# loss, accuracy = model.evaluate(testing_images, testing_labels)
#
# print(f"Loss: {loss}")
# print(f"Accuracy: {accuracy}")
#
# model.save('image_classifier.keras')


# After saving the trained model
model = models.load_model('image_classifier.keras')

# Function to get user input for the image
def get_user_input_image():
    image_path = input("Please enter the path of your image: ")
    img = cv.imread(image_path)
    if img is None:
        print("Error: Image not found. Please provide a valid path.")
        return None
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (32, 32))
    return img

# Get the user input image
img = get_user_input_image()
if img is not None:
    plt.imshow(img)
    plt.show()

    # Make prediction
    prediction = model.predict(np.array([img]) / 255)
    index = np.argmax(prediction)

    print(f'Prediction is {class_names[index]}')
