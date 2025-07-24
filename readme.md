# 🧠 CIFAR-10 Image Classification using VGG-style CNN with Dropout

This repo contain a deep learning project built with **Keras and TensorFlow**, where a VGG-like **Convolutional Neural Network (CNN)** is trained on the **CIFAR-10** dataset. It includes regularization using **Dropout** and demonstrates model persistence, visualization of results, and an interactive **Streamlit app** that classifies uploaded images.

---

## 📦 Dataset

We use the built-in `cifar10` dataset:
- 60,000 32×32 color images in 10 classes
- 50,000 training and 10,000 test images
- Classes: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`

---

## 🏗️ Project Structure

- **Model Implementation**: A VGG-like CNN with three convolutional blocks, followed by dense layers.
- **Dropout Regularization**: Introduced after pooling layers and before the final dense layer to combat overfitting.
- **Training & Evaluation**: 31 epochs, tracked training/validation metrics.
- **Visualization**: Accuracy/loss plots and a confusion matrix heatmap.
- **Streamlit App**: A frontend interface to upload and classify new images with the trained model.

---

## 🚀 Getting Started

### 1. Clone this repo
```bash
git clone https://github.com/Shankar7318/cnn-classifier.git
cd cnn-classifier
# 🚀 CIFAR-10 CNN Classifier — Streamlit App Setup

---

## 🧪 Install Dependencies

Make sure you’re using a Python 3.8+ environment. Then run:

```bash
pip install streamlit tensorflow pillow matplotlib seaborn numpy

# 💻 Launch the Streamlit App

streamlit run app.py

#📂 Project File Layout 
cnn-cifar10-classifier/
├── app.py
├── model_dropout.keras
├── README.md
├── training_notebook.ipynb
