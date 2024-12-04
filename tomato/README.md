# 🍅 Tomato Leaf Disease Classification with InceptionV3

This project implements a machine learning pipeline to classify tomato leaf diseases using the **InceptionV3** pre-trained model as a feature extractor. The pipeline incorporates data augmentation, transfer learning, and fine-tuning to achieve high accuracy. The trained model is available for download and can be deployed using Flask.

## 📂 Notebook and Model Access
- **Notebook**: [Tomato_Inception_V3_Final.ipynb](https://github.com/AgriScan-C242-PS521/Machine_Learning/blob/main/tomato/Tomato_InceptionV3_Final.ipynb)
- **Trained Model**: [Download from Google Drive](https://drive.google.com/file/d/1FMc3VgxUOLFTTyOssKnuA0svQq2tZtJ4/view?usp=sharing)

## 🏷️ Disease Labels
The model classifies images into the following 10 classes:
1. Tomato Mosaic Virus
2. Target Spot
3. Healthy
4. Bacterial Spot
5. Spider Mites (Two-spotted Spider Mite)
6. Leaf Mold
7. Septoria Leaf Spot
8. Late Blight
9. Early Blight
10. Tomato Yellow Leaf Curl Virus

**Final Accuracy**: 91%

---

## 🚀 Features
- **Transfer Learning**: Utilized the pre-trained **InceptionV3** model trained on ImageNet for feature extraction.
- **Data Augmentation**: Enhanced model robustness with techniques like random flipping, rotation, zooming, and more.
- **Early Stopping**: Prevented overfitting during training by using an early stopping callback.
- **Model Drive**: The trained model is available for download and deployment.

---

## 📂 Dataset
- **Source**: [Kaggle's Tomato Leaf Disease Dataset](https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf/data).
- The dataset contains images categorized into 10 classes representing healthy and diseased tomato leaves.
- **Splits**: Training, validation, and test sets.

---

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
- **Base Model**: InceptionV3 (weights pre-trained on ImageNet).
- Added custom layers for classification:
  - Fully connected layers with ReLU activations.
  - Batch Normalization for faster convergence.
  - Dropout for regularization.
  - Final softmax layer for multi-class classification.

### 4. **Training**
- Optimizer: **Adam**
- Loss function: **Categorical Crossentropy**
- Metric: **Accuracy**

### 5. **Model Saving**
- Saved the trained model in **TensorFlow SavedModel format** for Flask deployment.

---

## 🧪 Results
- **Validation Accuracy**: Achieved ~90% accuracy on the validation set.
- Fine-tuning improved performance further to reach 91% final accuracy.

---

## 🖥️ Deployment
To deploy the model using Flask:

1. **Download the trained model** from [Google Drive](https://drive.google.com/file/d/1FMc3VgxUOLFTTyOssKnuA0svQq2tZtJ4/view?usp=sharing).
2. Set up a Flask application and integrate the model for inference.

### Example Flask Code
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

---

## 📋 Requirements
- **Python 3.7+**
- **TensorFlow 2.8+**
- **Flask** (For deployment)

---

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

---

## 📜 Usage
1. **Download the model** from [Google Drive](https://drive.google.com/file/d/1FMc3VgxUOLFTTyOssKnuA0svQq2tZtJ4/view?usp=sharing).
2. **Set up Flask** for model deployment (refer to the `app.py` file).
3. Start the Flask server:
   ```bash
   python app.py
   ```

---

## ✨ Future Work
- Expand support to include more plant species and diseases.
- Improve deployment with a user-friendly web interface.

---

## 🤝 Contributions
Contributions, issues, and feature requests are welcome! Feel free to fork the repository and submit pull requests.

---

## 📄 License
This project is licensed under the MIT License. See the `LICENSE` file for details.

