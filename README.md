# 🌐 Language Detector Model

This project is a **Language Detection Model** built in Python.  
It loads and cleans a multilingual text dataset, trains a machine learning model, and allows users to test language predictions on custom input.

The model uses **TF-IDF Vectorization** and **Logistic Regression** to classify text into its respective language.  
Additionally, the project includes data cleaning, model evaluation, and visualization of results.

---

## 📂 Project Structure

Language-Detector-Model/
│
├── Dataset/
│ └── sentences_full_language.csv # Main dataset
│
├── Model/
│ └── model.pkl # Saved trained model
│
├── Cleaner_DS.py # Dataset loading & cleaning logic
├── Train_Model.py # Model training and evaluation
├── Test_Model.py # Model testing with user input
├── Confusion_Matrix.png # Confusion matrix heatmap
├── True_Positives_Line_Plot.png # True positives per class (line chart)
├── True_Positives_Histogram.png # True positives per class (histogram)
└── README.md # Project documentation

---

## ⚙️ Features

- **Dataset Analysis & Cleaning**
  - Loads CSV dataset
  - Removes repeated text entries (>3 times)
  - Filters languages with very low frequency

- **Model Training**
  - TF-IDF character-level features
  - Logistic Regression classifier
  - Accuracy score and classification report
  - Saves trained model as `.pkl`

- **Visualization**
  - Confusion matrix heatmap
  - True positives line plot
  - True positives histogram

- **Model Testing**
  - Interactive user input
  - Predicts language instantly

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/AUX-441/Language-Detector-Model.git
   cd Language-Detector-Model
