# 🌼 Flower Classification using Transfer Learning

This project demonstrates the use of **transfer learning** to classify flower images into five categories using both **TensorFlow** and **PyTorch** frameworks. It fine-tunes the pre-trained **ResNet50** model on a flower dataset to achieve high accuracy with minimal training.

## 🚀 Project Overview

- 🔍 Built for a deep learning class project at Pace University (CS672 - Spring 2025)
- 🧠 Combines **TensorFlow** and **PyTorch** implementations
- 📷 Utilizes transfer learning with **ResNet50**
- 🌺 Classifies flower images into 5 types: *Daisy, Dandelion, Rose, Sunflower, Tulip*
- 📊 Evaluates performance using accuracy, precision, recall, and F1-score

## 📂 Dataset

The ["Flowers Recognition" dataset](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition) contains **4,317 images** across five flower categories:
- Daisy 🌼
- Dandelion 🌾
- Rose 🌹
- Sunflower 🌻
- Tulip 🌷

Each class has approximately 800 images. The dataset is split into 75% training and 25% testing.

## 🧠 Model Architecture

### ResNet50 with Custom Classifier
- Loaded **ResNet50** without its top layer
- Added custom dense layers for classification
- Applied **dropout** to reduce overfitting
- **Froze** base layers for efficient training
- Fine-tuned top layers for improved performance

### Implemented in:
- ✅ TensorFlow (Keras API)
- ✅ PyTorch (`torchvision.models`)

## ⚙️ Training Details

- Optimizer: Adam
- Loss Function: CrossEntropyLoss
- Epochs: 20
- Metrics: Accuracy, Precision, Recall, F1-score
- Data Augmentation: Flip, Rotation, Zoom

## 📈 Evaluation Metrics

| Metric     | Score (example) |
|------------|-----------------|
| Accuracy   | 94.2%           |
| Precision  | 93.8%           |
| Recall     | 93.5%           |
| F1-Score   | 93.6%           |


## 📷 Visual Results

Include confusion matrix and sample predictions:

![Confusion Matrix](model_visualizations/confusion_matrix.png)
![Sample Predictions](examples/sample_predictions.png)

## 📦 Dependencies

Install via `requirements.txt`, or individually:

```bash
pip install tensorflow torch torchvision scikit-learn matplotlib pandas opencv-python
````

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/flower-classification-transfer-learning.git
cd flower-classification-transfer-learning
pip install -r requirements.txt
```

## 📁 Project Structure

```
.
├── flower_classifier.ipynb         # Jupyter Notebook (main project)
├── README.md
├── requirements.txt
├── model_visualizations/           # Confusion matrix, training curves
├── examples/                       # Example predictions
├── saved_model/                    # (Optional) Exported model files
└── .gitignore
```

## 💡 Key Concepts

* 📚 Transfer Learning
* 📸 Image Classification
* 🔁 Data Augmentation
* 📊 Model Evaluation


## 📌 Future Improvements

* Deploy the model as a web app using Flask or Streamlit
* Extend to more flower categories or real-time prediction
* Experiment with other architectures (EfficientNet, VGG)

