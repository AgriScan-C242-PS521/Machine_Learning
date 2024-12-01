# 🍅 Tomato Leaf Disease Classification with InceptionV3

This project implements a machine learning pipeline to classify tomato leaf diseases using the **InceptionV3** pre-trained model as a feature extractor. The pipeline incorporates data augmentation, transfer learning, and fine-tuning to achieve high accuracy.

## 🚀 Features
- **Transfer Learning**: Used the pre-trained **InceptionV3** model trained on ImageNet as the backbone for feature extraction.
- **Data Augmentation**: Applied techniques like random flipping, rotation, zooming, and more to increase model robustness.
- **Fine-Tuning**: Tuned the higher layers of the InceptionV3 model to adapt it to the tomato disease classification task.
- **Early Stopping**: Implemented an early stopping callback to avoid overfitting during training.
- **TensorFlow.js Compatibility**: The model can be converted into a **TensorFlow.js** format for deployment in web applications.
- **Model Drive**: https://drive.google.com/drive/folders/1pPZo7dVBAIiHx1viBGb8v0uDwjPysfAt?usp=sharing

## 📂 Dataset
The dataset consists of tomato leaf images categorized into 10 classes, representing healthy and various disease states. The data is split into **training** and **validation** sets.

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
- Optimizer: **Adam** with a learning rate of `1e-4` (fine-tuning at `1e-5`).
- Loss function: **Categorical Crossentropy**.
- Metrics: **Accuracy**.
- Trained for 10 epochs initially, followed by fine-tuning for additional epochs.

### 5. **Model Saving**
- Saved the trained model in **TensorFlow SavedModel format** for compatibility with TensorFlow.js.

## 🧪 Results
- **Validation Accuracy**: Achieved a high accuracy (~90%) on the validation set.
- Fine-tuned the model to further improve performance.

## 🖥️ Deployment
To deploy the model on the web:
1. Convert the model to TensorFlow.js format using:
   ```bash
   tensorflowjs_converter --input_format=tf_saved_model \
   /path/to/saved_model /path/to/tfjs_model
   ```
2. Integrate the converted model into a web application.

## 📋 Requirements
- Python 3.7+
- TensorFlow 2.8+
- TensorFlow.js (for model conversion)
- Libraries: `numpy`, `matplotlib`, `seaborn`

## 📁 Folder Structure
```
.
├── data/
│   ├── train/
│   ├── val/
├── notebooks/
│   ├── training.ipynb
├── models/
│   ├── Tomato_InceptionV3_V1_2/  # SavedModel format
├── README.md
```

## 📜 Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/tomato-leaf-disease-classification.git
   cd tomato-leaf-disease-classification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model:
   - Use the Jupyter notebook in `notebooks/training.ipynb`.
4. Convert the model to TensorFlow.js:
   ```bash
   tensorflowjs_converter --input_format=tf_saved_model \
   /path/to/saved_model /path/to/tfjs_model
   ```

## ✨ Future Work
- Add support for more plant species and diseases.
- Improve deployment with a user-friendly web interface.
- Experiment with other architectures like EfficientNet.

## 🤝 Contributions
Contributions, issues, and feature requests are welcome! Feel free to fork the repository and submit pull requests.

## 📄 License
This project is licensed under the MIT License. See the `LICENSE` file for details.
