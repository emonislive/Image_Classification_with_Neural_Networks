import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from keras import datasets, models, layers
from keras.optimizers import Adam, SGD, RMSprop, Adagrad
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
import threading
import keras.callbacks as cb

# ? training and testing data display format [DATASET CIFAR-10]
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
# ? preparing the data by assigning pixels to 0-1 out of 255 (scaling down the data)
training_images, testing_images = training_images / 255.0, testing_images / 255.0

# ! the order of the objects in the dataset [DATASET CIFAR-10]
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Reduce the data size (optional)
training_images = training_images[:40000]
training_labels = training_labels[:40000]
testing_images = testing_images[:6000]
testing_labels = testing_labels[:6000]

# ----------------- MODEL LOADING -----------------
# Load a pre-trained model (if it exists)
model = models.load_model('image_classifier.keras')

# ----------------- UTILITY FUNCTIONS -----------------
def img_to_tk(img):
    """
    Convert a NumPy image (RGB) to a Tkinter-compatible image.
    If the image is float [0-1], convert it to uint8 [0-255].
    """
    if img.dtype != 'uint8':
        img = (img * 255).astype('uint8')
    pil_img = Image.fromarray(img)
    return ImageTk.PhotoImage(pil_img)

def get_user_input_image():
    """
    Open a file dialog for the user to select an image.
    Return a 32x32 RGB image (NumPy array) or None if invalid/cancelled.
    """
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[
            ("Image Files", ("*.png", "*.jpg", "*.jpeg", "*.bmp")),
            ("All Files", "*.*")
        ]
    )
    if not file_path:
        return None

    img = cv.imread(file_path)
    if img is None:
        return None

    # Convert from BGR to RGB, resize to 32x32 for our CIFAR-10 model
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (32, 32))
    return img

# ----------------- CLASSIFICATION (HOME) -----------------
def make_prediction():
    """
    Ask user for an image, display it, and show the prediction result.
    Also display dummy accuracy and training parameters.
    """
    img = get_user_input_image()
    if img is not None:
        # Update preview image
        tk_img = img_to_tk(img)
        preview_label.config(image=tk_img, text="")
        preview_label.image = tk_img

        # Predict (scale to [0-1])
        prediction = model.predict(np.array([img]) / 255.0)
        index = np.argmax(prediction)
        class_name = class_names[index]
        result_label.config(text=f"Prediction: {class_name}")

        # Display example accuracy/params (replace with real values if you have them)
        accuracy_label.config(text="Accuracy: 90.7%")
        params_label.config(text="Optimizer: Adam\nLoss: Sparse Categorical Crossentropy\nEpochs: 20")

# ----------------- TESTING DATASET PREVIEW -----------------
def load_testing_dataset():
    """
    Display the first 16 images from the testing dataset in a 4x4 grid inside a scrollable frame.
    """
    # Clear existing widgets in test_images_frame
    for widget in test_images_frame.winfo_children():
        widget.destroy()

    # Show 16 images in a 4x4 grid
    for i in range(16):
        img = testing_images[i]
        # Resize to 100x100 for preview
        img_resized = cv.resize(img, (100, 100))
        tk_img = img_to_tk(img_resized)

        lbl = ttk.Label(test_images_frame, image=tk_img)
        lbl.image = tk_img  # keep a reference
        row, col = divmod(i, 4)
        lbl.grid(row=row, column=col, padx=5, pady=5)

def load_testing_dataset_async():
    """
    Run load_testing_dataset in a thread to avoid blocking the GUI.
    """
    threading.Thread(target=load_testing_dataset, daemon=True).start()

# ----------------- TRAINING CALLBACK -----------------
class TrainLogger(cb.Callback):
    """
    Custom Keras callback to log training progress to a Tkinter text widget.
    """
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epochs = self.params.get('epochs', 0)
        msg = (
            f"Epoch {epoch+1}/{epochs} "
            f"- loss: {logs.get('loss',0.0):.4f} "
            f"- accuracy: {logs.get('accuracy',0.0):.4f} "
            f"- val_loss: {logs.get('val_loss',0.0):.4f} "
            f"- val_accuracy: {logs.get('val_accuracy',0.0):.4f}\n"
        )
        self.text_widget.config(state='normal')
        self.text_widget.insert(END, msg)
        self.text_widget.see(END)
        self.text_widget.update_idletasks()
        self.text_widget.config(state='disabled')

# ----------------- TRAINING WITH TWEAKABLE OPTIONS -----------------
def get_training_params():
    """
    Read user-specified hyperparameters from the UI.
    """
    try:
        e = int(epochs_var.get())
    except:
        e = 5
    try:
        b = int(batch_var.get())
    except:
        b = 32
    try:
        lr = float(lr_var.get())
    except:
        lr = 0.001

    opt_choice = opt_var.get()
    # Map optimizer names to actual optimizer instances
    if opt_choice == "Adam":
        optimizer = Adam(learning_rate=lr)
    elif opt_choice == "SGD":
        optimizer = SGD(learning_rate=lr)
    elif opt_choice == "RMSProp":
        optimizer = RMSprop(learning_rate=lr)
    elif opt_choice == "Adagrad":
        optimizer = Adagrad(learning_rate=lr)
    else:
        optimizer = Adam(learning_rate=lr)

    return e, b, optimizer

def start_training():
    """
    Build and train a new CNN model with user-chosen hyperparameters.
    Show progress in the train_text widget.
    Then save the model and update the global `model` reference.
    """
    global model  # so we can replace the old model after training

    # Clear the text box and enable it for writing
    train_text.config(state='normal')
    train_text.delete('1.0', END)
    train_text.config(state='disabled')

    epochs, batch_size, optimizer = get_training_params()

    # Build a fresh model (similar architecture)
    new_model = models.Sequential([
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

    new_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Let the user know we're starting
    train_text.config(state='normal')
    train_text.insert(END, f"Starting training...\n")
    train_text.see(END)
    train_text.config(state='disabled')

    # Train
    new_model.fit(
        training_images,
        training_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(testing_images, testing_labels),
        callbacks=[TrainLogger(train_text)]
    )

    # Evaluate
    loss, accuracy = new_model.evaluate(testing_images, testing_labels)
    train_text.config(state='normal')
    train_text.insert(END, f"\nTraining complete.\nFinal Loss: {loss:.4f}, Accuracy: {accuracy:.4f}\n")
    train_text.insert(END, "Saving model as 'image_classifier.keras'...\n")
    train_text.see(END)
    train_text.config(state='disabled')

    # Save the new model
    new_model.save('image_classifier.keras')

    # Update the global reference so classification uses the newly trained model
    model = new_model

# ----------------- GUI SETUP -----------------
root = ttk.Window(themename="cyborg")
root.title("CIFAR-10 Image Classifier")
root.geometry("1200x800")

# Define bigger custom style overrides for fonts and color
style = ttk.Style()
style.configure("HugeTitle.TLabel", font=("Helvetica", 36, "bold"))
style.configure("LargeSection.TLabel", font=("Helvetica", 24, "bold"))
style.configure("Large.TLabel", font=("Helvetica", 18))
style.configure("Large.TButton", font=("Helvetica", 18, "bold"))
style.configure("Medium.TButton", font=("Helvetica", 16, "bold"))

# Notebook for tabbed interface
notebook = ttk.Notebook(root, bootstyle=PRIMARY)
notebook.pack(expand=True, fill="both", padx=10, pady=10)

# Create tabs
home_tab = ttk.Frame(notebook)
testing_tab = ttk.Frame(notebook)
training_tab = ttk.Frame(notebook)
instructions_tab = ttk.Frame(notebook)

notebook.add(home_tab, text="Home")
notebook.add(testing_tab, text="Testing Dataset")
notebook.add(training_tab, text="Train Model")
notebook.add(instructions_tab, text="Instructions")

# ----------------- HOME TAB -----------------
home_title = ttk.Label(
    home_tab,
    text="Welcome to the CIFAR-10 Image Classifier!",
    style="HugeTitle.TLabel",
    bootstyle="inverse-info",
    anchor="center"
)
home_title.pack(pady=20)

home_info = """This application demonstrates a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset.
You can use the buttons below to load your own image and see how the model classifies it.
Navigate to the other tabs for more features:

- Testing Dataset: Preview some images from the CIFAR-10 testing set.
- Train Model: Retrain or fine-tune the CNN with different hyperparameters.
- Instructions: Learn how the code is structured and what each part does.
"""

info_label = ttk.Label(
    home_tab,
    text=home_info,
    style="Large.TLabel",
    bootstyle="inverse-secondary",
    wraplength=900,
    justify="left"
)
info_label.pack(pady=20)

insert_button = ttk.Button(
    home_tab,
    text="Insert Image for Classification",
    style="Large.TButton",
    bootstyle="success",
    command=make_prediction
)
insert_button.pack(pady=10)

preview_label = ttk.Label(
    home_tab,
    text="No Image Selected",
    style="Large.TLabel",
    bootstyle="inverse-secondary"
)
preview_label.pack(pady=20)

result_label = ttk.Label(
    home_tab,
    text="Prediction: None",
    style="LargeSection.TLabel",
    bootstyle="inverse-success",
    anchor="center"
)
result_label.pack(pady=10)

accuracy_label = ttk.Label(
    home_tab,
    text="Accuracy: None",
    style="Large.TLabel",
    bootstyle="inverse-warning"
)
accuracy_label.pack(pady=5)

params_label = ttk.Label(
    home_tab,
    text="Training Parameters: None",
    style="Large.TLabel",
    bootstyle="inverse-danger"
)
params_label.pack(pady=5)

# ----------------- TESTING DATASET TAB -----------------
canvas = ttk.Canvas(testing_tab)
canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)

scrollbar = ttk.Scrollbar(testing_tab, orient="vertical", command=canvas.yview)
scrollbar.pack(side="right", fill="y")

canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

test_images_frame = ttk.Frame(canvas)
canvas.create_window((0, 0), window=test_images_frame, anchor="nw")

load_test_btn = ttk.Button(
    testing_tab,
    text="Load Testing Dataset",
    style="Large.TButton",
    bootstyle="info",
    command=load_testing_dataset_async
)
load_test_btn.pack(pady=10)

# ----------------- TRAINING TAB -----------------
train_title_label = ttk.Label(
    training_tab,
    text="Train the CIFAR-10 Model",
    style="HugeTitle.TLabel",
    bootstyle="inverse-info",
    anchor="center"
)
train_title_label.pack(pady=20)

# Frame for hyperparameter inputs
param_frame = ttk.Frame(training_tab)
param_frame.pack(pady=10)

# Variables to hold user input
epochs_var = ttk.StringVar(value="5")
batch_var = ttk.StringVar(value="32")
lr_var = ttk.StringVar(value="0.001")
opt_var = ttk.StringVar(value="Adam")

# EPOCHS
ttk.Label(param_frame, text="Epochs:", style="Large.TLabel").grid(row=0, column=0, padx=5, pady=5, sticky="e")
epochs_entry = ttk.Entry(param_frame, textvariable=epochs_var, width=5, font=("Helvetica", 18))
epochs_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

# BATCH SIZE
ttk.Label(param_frame, text="Batch Size:", style="Large.TLabel").grid(row=1, column=0, padx=5, pady=5, sticky="e")
batch_entry = ttk.Entry(param_frame, textvariable=batch_var, width=5, font=("Helvetica", 18))
batch_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")

# LEARNING RATE
ttk.Label(param_frame, text="Learning Rate:", style="Large.TLabel").grid(row=2, column=0, padx=5, pady=5, sticky="e")
lr_entry = ttk.Entry(param_frame, textvariable=lr_var, width=6, font=("Helvetica", 18))
lr_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")

# OPTIMIZER
ttk.Label(param_frame, text="Optimizer:", style="Large.TLabel").grid(row=3, column=0, padx=5, pady=5, sticky="e")
opt_combo = ttk.Combobox(param_frame, textvariable=opt_var, values=["Adam", "SGD", "RMSProp", "Adagrad"], font=("Helvetica", 18), state="readonly")
opt_combo.grid(row=3, column=1, padx=5, pady=5, sticky="w")

start_train_button = ttk.Button(
    training_tab,
    text="Start Training",
    style="Large.TButton",
    bootstyle="success",
    command=lambda: threading.Thread(target=start_training, daemon=True).start()
)
start_train_button.pack(pady=10)

train_text = ScrolledText(training_tab, width=80, height=15, font=("Consolas", 16))
train_text.pack(padx=10, pady=10)
train_text.config(state='disabled')  # read-only by default

# ----------------- INSTRUCTIONS TAB -----------------
instructions_label = ttk.Label(
    instructions_tab,
    text="Instructions and Code Overview",
    style="HugeTitle.TLabel",
    bootstyle="inverse-info",
    anchor="center"
)
instructions_label.pack(pady=20)

instructions_text = ScrolledText(instructions_tab, width=100, height=20, font=("Helvetica", 16))
instructions_text.pack(padx=10, pady=10)

# Fill in instructions explaining each part of the code
instructions_content = """\
HOW THIS MODEL WORKS
--------------------
1) Data Loading and Preprocessing:
   - We use Keras to load the CIFAR-10 dataset.
   - Images are scaled from [0,255] to [0,1].

2) Model Architecture:
   - A simple CNN with 3 convolutional layers, each followed by max pooling.
   - Flatten, then two Dense layers (64 units, then 10 outputs with softmax).

3) Training:
   - In the 'Train Model' tab, you can adjust epochs, batch size, learning rate, and optimizer.
   - When you click 'Start Training', a new model is built and trained on the CIFAR-10 dataset.
   - Real-time logs appear in the text box, showing epoch-wise loss and accuracy.
   - After training, the model is saved as 'image_classifier.keras' and replaces the globally loaded model.

4) Classification (Home Tab):
   - Click 'Insert Image for Classification' to load an image from your system.
   - The model resizes it to 32x32 and predicts which CIFAR-10 class it belongs to.
   - The predicted class is displayed along with example accuracy/params.

5) Testing Dataset Tab:
   - Click 'Load Testing Dataset' to see 16 images from the CIFAR-10 testing set in a scrollable frame.

6) Code Structure:
   - 'TrainLogger' callback logs training progress to the ScrolledText widget.
   - 'make_prediction' handles loading a user image and feeding it to the model.
   - 'start_training' handles building and compiling the CNN, then training it with user-selected hyperparameters.
   - The UI is built using ttkbootstrap with the 'cyborg' theme for a modern look.

Enjoy exploring and customizing the CIFAR-10 Image Classifier!
"""

instructions_text.config(state='normal')
instructions_text.insert(END, instructions_content)
instructions_text.config(state='disabled')

# ----------------- COMMENTED DATASET PREVIEW (Matplotlib) -----------------
# for i in range(16):
#     plt.subplot(4,4, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(training_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[training_labels[i][0]])
# plt.show()

# ----------------- MAINLOOP -----------------
root.mainloop()
