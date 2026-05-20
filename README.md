# Tomato-Disease-Detection-
The goal of this project was to build a robust tomato leaf disease detection system that performed effectively in real agricultural environments. Unlike many existing studies that relied only on lab-controlled datasets such as PlantVillage, this project focused on real-world farmer images containing noisy backgrounds, varying lighting conditions, shadows, and complex field environments.

The models were trained on a hybrid dataset containing both laboratory and real-life field images, but the final evaluation was performed only on a separate testing dataset made up of real farmer-style images.

## Model Performance on Real-World Test Dataset
Custom CNN: 86% Accuracy
VGG16: 44% Accuracy
ResNet50: 38% Accuracy

The results showed that the Custom CNN generalized much better to real-field conditions, while transfer learning models struggled because of domain mismatch and background complexity.

🧠 Models Used
🔹 Custom CNN

A task-specific CNN architecture designed for noisy real-field images.

Layers Used:
Convolution Layers: 32 → 64 → 128 → 256 → 512 filters
Kernel Size: 3×3
Max Pooling Layers
Batch Normalization
Dropout Layers (0.5 and 0.3)
Dense Layers (512 and 256 neurons)
Softmax Output Layer (10 Classes)
Techniques Used:
Adam Optimizer
Learning Rate Fine-Tuning
Data Augmentation
Regularization to reduce overfitting
🔹 VGG16 (Transfer Learning)

A pre-trained CNN model originally trained on ImageNet and fine-tuned for tomato disease classification.

Layers & Modifications:
Pre-trained VGG16 Base
Data Augmentation Layers:
Random Flip
Random Rotation
Additional Conv2D Layers:
1024 filters
Max Pooling
Flatten Layer
Dense Layers:
1024 → 512 → 256 neurons
Dropout Layers (0.5)
Softmax Output Layer
Techniques Used:
Transfer Learning
SGD Optimizer with Momentum
Fine-Tuning
🔹 ResNet50 (Transfer Learning)

A deep residual learning model capable of learning complex visual patterns.

Layers Used:
Pre-trained ResNet50 Base
Global Average Pooling Layer
Flatten Layer
Dropout Layer (0.2)
Dense Softmax Output Layer (10 Classes)
Techniques Used:
Residual Connections
Adam Optimizer
Transfer Learning
Feature Extraction
🌐 Web Application

A web application was developed using Flask that allowed users to:

Upload tomato leaf images
Receive disease prediction with confidence score
View mapped treatment recommendations
Get a warning if prediction confidence was below 60%

The application was designed with a simple and farmer-friendly interface so it could be easily used in practical agricultural environments.

🗂 Dataset

The dataset contained:

PlantVillage images
PlantDoc images
TomatoVillage images
Manually scraped farmer-style images
Total Dataset Size:

16,570 Images

Classes:
Bacterial Spot
Early Blight
Late Blight
Leaf Mold
Septoria Leaf Spot
Target Spot
Two-Spotted Spider Mite
Tomato Yellow Leaf Curl Virus
Tomato Mosaic Virus
Healthy
🔧 Technologies Used
Python
TensorFlow / Keras
Flask
OpenCV
NumPy
HTML/CSS
