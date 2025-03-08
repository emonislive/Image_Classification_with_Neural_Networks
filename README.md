# Image Classification with CIFAR-10

A deep learning project for image classification using CIFAR-10 dataset with a PyQt5 GUI interface.

![GUI Preview](gui-preview.png)

## Features

- **CNN Model**: Custom convolutional neural network
- **GUI Interface**: User-friendly graphical interface
- **Dataset Preview**: 4x4 grid of CIFAR-10 samples
- **Training Controls**: Customizable training parameters
- **Image Prediction**: Upload and classify custom images

## Requirements

### System Requirements used to train the model
- Windows 11 (64-bit)
- Python 3.8-3.10
- 16GB RAM
- AMD RX 580/NVIDIA GPU (Optional but recommended)

### Python Packages
```bash
# requirements.txt
numpy==1.24.4
tensorflow==2.13.0
keras==2.13.1
opencv-python==4.9.0.80
matplotlib==3.7.5
scipy==1.10.1
PyQt5==5.15.10
pygments==2.17.2
Installation
```
#### Clone repository:
```bash
git clone https://github.com/yourusername/image-classification.git
cd image-classification
```
#### Create and activate virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate
```
#### Install dependencies:
```bash
pip install -r requirements.txt
```
## Usage
#### Training the Model
```bash
python pretraining_model.py
```
#### Running the GUI
```bash
python main_gui.py
```
### Model Architecture
```python
Sequential([
    layers.Input(shape=(32,32,3)), 
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```
