# 🐱🐶 Cat vs Dog Image Classifier
This project is a beginner-friendly deep learning classifier that distinguishes between cats and dogs using a Convolutional Neural Network (CNN) built with TensorFlow and Keras.

---

## 💡 What does this project include?

- Dataset preparation (train/validation split)
- Image preprocessing and normalization
- CNN model design and training
- Model saving and prediction on new images
- Training graphs for accuracy and loss

---

## 🧠 Technologies Used

- Python 3
- TensorFlow / Keras
- Matplotlib
- OS / glob (for file management)

## 🗂️ Folder Structure

---

## 🚀 How to Run

1. Install the required packages:

```bash
pip install -r requirements.txt

## 🐱🐶 Dataset Setup

Due to dataset size, the image files are not included in this repository.

### 🔽 To use the project:
1. Download the **Asirra Cats vs Dogs** dataset (or your own cat/dog image dataset).
2. Place the images in a folder named:
> 📁 The folder should contain images in `.jpg` or `.png` format.  
> You can use any folder name you prefer — just make sure to update the path in the script.

3. Run `data_distribution.py` to split the data into training and validation folders.

## 📁 Directory Structure (example)

project/ │
├── Asirra_ cat vs dogs/ # <- Your raw image dataset
  ├── train/
     │├── cats/
     |└── dogs/
  ├── val/
     |├── cats/
     │└── dogs/
├── catdog_classifier_train.py └── ...


Train the model:
python catdog_classifier_train.py
Make a prediction on a new image:
python prediction.py
View training graphs:
python graph_drawing.py

# 📸 Example Prediction
After training, you can add an image inside the predict/ folder named my_test_image.jpg, and run the prediction script to see whether it's a cat or a dog 🐾

👩‍💻 Created By
Sara Farkash – first steps into the exciting world of AI ❤️
