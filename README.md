# cnn-image-prediction
Image classification with CNN

# CNN with VOC2012 – Multi-Label Image Classification

This project implements a **Convolutional Neural Network (CNN)** for **multi-label image classification** using the [PASCAL VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) dataset.

## 🔍 Overview

- Uses XML annotation files to extract image labels
- Performs multi-label classification (e.g., an image can contain both “dog” and “person”)
- Built with **TensorFlow/Keras**
- Simple CNN architecture suitable for educational and research purposes

## 🗂️ Dataset Structure
  VOC2012/
  
  ├── Annotations/
  
  ├── JPEGImages/
  
  ├── ImageSets/
  
Ensure the dataset is extracted and paths are correctly defined in the script.

## 🚀 Getting Started

1. Clone this repository:
bash
git clone https://github.com/yourusername/voc2012-cnn.git
cd voc2012-cnn

pip install -r requirements.txt

python train_model.py

🧠 Model Info
CNN with 2 Conv2D + MaxPooling layers

Binary cross-entropy loss

Final output layer uses sigmoid activation
Model is saved as:
  model.keras

📈 Results
Training and validation accuracy/loss are plotted and saved:

accuracy.png

loss.png

🌐 Portfolio

This project is also presented on my personal portfolio website:

🔗 [Visit Site](https://cancrpcoglu.github.io/website/)


👤 Author
Can Çorapçıoğlu

Final Year Computer Engineering Student, Atılım University

[LinkedIn](https://www.linkedin.com/in/can-%C3%A7orap%C3%A7%C4%B1o%C4%9Flu-15a340247/)
