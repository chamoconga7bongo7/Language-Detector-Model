# 🌐 Language Detector Model

This project is a **Language Detection Model** built in Python.  
It loads and cleans a multilingual text dataset, trains a machine learning model, and allows users to test language predictions on custom input.

The model uses **TF-IDF Vectorization** and **Logistic Regression** to classify text into its respective language.  
Additionally, the project includes data cleaning, model evaluation, and visualization of results.

---

## ✨ Features

🚀 **Fast Language Detection**  
Detects the language of any given text within milliseconds using a trained ML model.

🧹 **Automated Data Cleaning**  
Removes duplicate or over-represented text entries to improve model accuracy.

📊 **Model Evaluation & Reporting**  
Generates accuracy score, classification report, and detailed confusion matrix.

🎨 **Beautiful Visualizations**  
Creates heatmaps, histograms, and line plots to visualize model performance.

💾 **Model Persistence**  
Saves the trained model as a `.pkl` file for easy re-use without retraining.

🖥 **Interactive Testing**  
Accepts user input directly from the terminal to test predictions instantly.

🛠 **Customizable Training Pipeline**  
Easily adjust vectorization method, features, and classifier for experimentation.

📁 **Multi-Language Dataset Support**  
Handles large multilingual datasets with millions of entries efficiently.


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
