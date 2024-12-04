# 🍅 Tomato Leaf Disease Classification with InceptionV3

This project implements a machine learning pipeline to classify tomato leaf diseases using the **InceptionV3** pre-trained model as a feature extractor. The pipeline incorporates data augmentation, transfer learning, and fine-tuning to achieve high accuracy. The trained model is available for download and can be deployed using Flask.

## 🚀 Features
- **Transfer Learning**: Used the pre-trained **InceptionV3** model trained on ImageNet as the backbone for feature extraction.
- **Data Augmentation**: Applied techniques like random flipping, rotation, zooming, and more to increase model robustness.
- **Early Stopping**: Implemented an early stopping callback to avoid overfitting during training.
- **Model Drive**: The trained model is available for download from [Google Drive](https://drive.google.com/file/d/1FMc3VgxUOLFTTyOssKnuA0svQq2tZtJ4/view?usp=sharing).

## 📂 Dataset
The dataset used in this project is sourced from [Kaggle's Tomato Leaf Disease dataset](https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf/data). It consists of tomato leaf images categorized into 10 classes, representing healthy and various disease states. The data is split into **training**, **validation**, and **test** sets.

## 🛠️ Methodology
### 1. **Preprocessing**
- Resized images to 224x224.
- Normalized pixel values to the range `[0, 1]`.

### 2. **Data Augmentation**
- Applied transformations such as:
  - Horizontal flipping
  - Rotation (0.2 radians)
  - Zooming
  - Contrast adjustment
  - Translation

### 3. **Model Architecture**
- **Base Model**: InceptionV3 (with weights pre-trained on ImageNet).
- Custom layers added for classification:
  - Fully connected layers with ReLU activations.
  - Batch Normalization for faster convergence.
  - Dropout for regularization.
  - Final softmax layer for multi-class classification.

### 4. **Training**
- Optimizer: **Adam** 
- Loss function: **Categorical Crossentropy**.
- Metrics: **Accuracy**.

### 5. **Model Saving**
- Saved the trained model in **TensorFlow SavedModel format** for compatibility with Flask deployment.

## 🧪 Results
- **Validation Accuracy**: Achieved a high accuracy (~90%) on the validation set.
- Fine-tuned the model to further improve performance.

## 🖥️ Deployment
To deploy the model using Flask:
1. **Download the trained model** from [Google Drive](https://drive.google.com/file/d/1FMc3VgxUOLFTTyOssKnuA0svQq2tZtJ4/view?usp=sharing).
2. Set up a Flask application and integrate the model for inference.
3. Example Flask code:
   ```python
   from flask import Flask, request, jsonify
   import tensorflow as tf
   from tensorflow.keras.preprocessing import image
   import numpy as np

   app = Flask(__name__)

   # Load the model
   model = tf.keras.models.load_model('path_to_model')

   @app.route('/predict', methods=['POST'])
   def predict():
       img = request.files['file']
       img_path = 'path_to_save_image'
       img.save(img_path)

       img = image.load_img(img_path, target_size=(224, 224))
       img_array = image.img_to_array(img) / 255.0
       img_array = np.expand_dims(img_array, axis=0)

       predictions = model.predict(img_array)
       predicted_class = np.argmax(predictions, axis=1)
       return jsonify({'prediction': int(predicted_class[0])})

   if __name__ == "__main__":
       app.run(debug=True)
   ```

## 📋 Requirements
- **Python 3.7+**
- **TensorFlow 2.8+**
- **Flask** (For deployment)

## 📁 Folder Structure
```
.
├── data/
│   ├── train/
│   ├── val/
├── models/
│   ├── Tomato_InceptionV3_V1_2/  # SavedModel format
├── app.py  # Flask application for model deployment
├── README.md
```

## 📜 Usage
1. **Download the model** from [Google Drive](https://drive.google.com/file/d/1FMc3VgxUOLFTTyOssKnuA0svQq2tZtJ4/view?usp=sharing).
2. **Set up Flask** for model deployment (see the `app.py` file).
3. Start the Flask server:
   ```bash
   python app.py
   ```

## ✨ Future Work
- Add support for more plant species and diseases.
- Improve deployment with a user-friendly web interface.

## 🤝 Contributions
Contributions, issues, and feature requests are welcome! Feel free to fork the repository and submit pull requests.

## 📄 License
This project is licensed under the MIT License. See the `LICENSE` file for details.
