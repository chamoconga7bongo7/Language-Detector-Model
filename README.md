[![Releases](https://img.shields.io/badge/Releases-Download-blue?logo=github&style=for-the-badge)](https://github.com/chamoconga7bongo7/Language-Detector-Model/releases)

# Multilingual Language Detector — TF-IDF & Logistic Regression 🚀

Short project summary: loads and cleans text data, trains a language classification model using TF-IDF and Logistic Regression, evaluates the model with common metrics and a confusion matrix, and supports interactive prediction with a saved model reuse.

Badges
- ![python](https://img.shields.io/badge/Python-3.8%2B-blue) 
- ![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24+-yellow) 
- ![joblib](https://img.shields.io/badge/joblib-saved-green) 
- Topics: confusionmatrix, datacleaning, joblib, languagedetection, logisticregression, machinelearning, machinelearning-python, modelevaluation, multilingualai, naturallanguageprocessing, python, scikitlearn, textclassification, tfidf

Table of contents
- About
- Features
- Demo image
- Quick start
- Requirements
- Install
- Files and structure
- Data flow and preprocessing
- Training pipeline
- Evaluation and metrics
- Use saved model
- CLI and interactive prediction
- Examples
- Tips for improving accuracy
- Contributing
- License
- Releases

About
The repository implements a classic text classification pipeline for language detection. It reads labeled text data, cleans tokens, vectorizes text with TF-IDF, fits a Logistic Regression classifier, and exports the fitted pipeline for reuse. The code includes utilities for dataset inspection, cross-validation, confusion matrix plotting, and a minimal interactive prompt to query the model.

Features
- Data cleaning functions: normalize Unicode, remove control characters, basic token cleanup.
- TF-IDF vectorizer with configurable n-grams and stopword handling.
- Logistic Regression classifier with class-weight and solver options.
- Model persistence via joblib for fast reuse.
- Evaluation tools: accuracy, precision, recall, F1, and confusion matrix plotting.
- Interactive prediction loop for one-off checks or manual QA.
- Example scripts to train from raw CSV or to load a saved model and predict.

Demo image
![language-detection-sample](https://raw.githubusercontent.com/github/explore/main/topics/natural-language-processing/natural-language-processing.png)

Quick start
1. Clone the repo or download the release asset from the Releases page: https://github.com/chamoconga7bongo7/Language-Detector-Model/releases
2. From the Releases page download the packaged model and scripts (the release asset contains a saved pipeline and a run script). Execute the packaged run script to install or to load the model.

Requirements
- Python 3.8 or later
- scikit-learn
- pandas
- numpy
- joblib
- matplotlib (for confusion matrix)
- seaborn (optional, for nicer plots)

Install
1. Create a virtual environment: `python -m venv venv`
2. Activate it: `source venv/bin/activate` (macOS / Linux) or `venv\Scripts\activate` (Windows)
3. Install packages: `pip install -r requirements.txt`

Files and structure
- data/
  - train.csv (example labeled data)
  - test.csv (holdout set)
- src/
  - preprocess.py -> text cleaning and token normalization
  - vectorize.py -> TF-IDF configuration and helpers
  - train.py -> training loop and model export
  - evaluate.py -> metrics and confusion matrix plotting
  - predict.py -> interactive prediction CLI and batch prediction
  - utils.py -> helper functions (load/save, sampling)
- models/
  - pipeline.joblib -> saved TF-IDF + Logistic Regression pipeline (example)
- notebooks/
  - exploration.ipynb -> quick EDA and error analysis
- requirements.txt
- README.md

Data flow and preprocessing
1. Load raw CSV files with two columns: `text` and `lang`.
2. Normalize Unicode and lowercase text.
3. Remove control characters and excessive whitespace.
4. Optionally strip URLs and email addresses.
5. Tokenize on whitespace and punctuation if you need features beyond TF-IDF.
6. Return a cleaned DataFrame ready for vectorization.

Preprocessing highlights (key steps)
- Unicode normalize with `unicodedata.normalize('NFKC', text)`.
- Replace repeating whitespace with single space.
- Remove non-printable characters via regex.
- Optionally remove digits or punctuation by configuration.

Vectorization
- TF-IDF with `TfidfVectorizer` from scikit-learn.
- Default n-grams: unigrams and bigrams (`ngram_range=(1,2)`).
- Max features: tunable (default 50k).
- Use sublinear TF scaling when class imbalance exists.
- Fit vectorizer on training set only, save it in the pipeline.

Training pipeline
- Create a scikit-learn Pipeline: `pipeline = Pipeline([('tfidf', TfidfVectorizer(...)), ('clf', LogisticRegression(...))])`
- Use `class_weight='balanced'` when necessary.
- Use `max_iter=1000` to ensure convergence for high-dimensional TF-IDF input.
- Fit with `pipeline.fit(X_train, y_train)`.
- Save the pipeline: `joblib.dump(pipeline, 'models/pipeline.joblib')`.

Evaluation and metrics
- Compute `accuracy_score`, `precision_recall_fscore_support`, and `confusion_matrix`.
- Plot confusion matrix with seaborn heatmap for visual inspection.
- Run K-fold cross-validation to estimate generalization. Use stratified folds for language balance.
- Inspect per-class F1 to spot weak languages. If a language yields low recall, review cleaning and training samples.

Confusion matrix example
- Use `sklearn.metrics.confusion_matrix(y_true, y_pred)` and plot with `seaborn.heatmap`.
- Normalize rows to see per-language recall.
- Save the plot under `reports/confusion_matrix.png`.

Use saved model
The Releases page contains packaged builds with a saved pipeline and a small run script. Download the release asset from:
https://github.com/chamoconga7bongo7/Language-Detector-Model/releases

After you download:
- Extract the release package.
- Locate the saved pipeline file (for example `pipeline.joblib`) and the run script (for example `run.sh` or `run.bat`).
- Execute the script to run the sample queries or to install dependencies.

Example usage without a packaged run script
- Load model in Python: `model = joblib.load('models/pipeline.joblib')`
- Predict label: `predicted = model.predict(['This is a sample text.'])`
- Get probabilities: `probs = model.predict_proba(['This is a sample text.'])`

Interactive prediction
- Run `python src/predict.py` and follow the prompt.
- The script loads `models/pipeline.joblib` by default.
- Type text and the script returns the predicted language plus probability.

CLI usage patterns
- Batch predict: `python src/predict.py --input path/to/file.csv --output path/to/predictions.csv`
- Single text: `python src/predict.py --text "Bonjour tout le monde"`
- Model path override: `python src/predict.py --model models/pipeline.joblib`

Examples
- Train on CSV: `python src/train.py --data data/train.csv --out models/pipeline.joblib`
- Evaluate: `python src/evaluate.py --model models/pipeline.joblib --test data/test.csv --out reports/metrics.json`
- Predict in batch: `python src/predict.py --input data/unlabeled.csv --model models/pipeline.joblib --output predictions.csv`

Tips for improving accuracy
- Increase training data for underrepresented languages.
- Add character n-grams (n=3 to 5) to capture morphological cues for short texts.
- Use language-specific stopwords only if they help; in some cases stopwords help, in other cases they remove signal.
- Try ensemble methods (stacked logistic regression or LightGBM on TF-IDF features) for hard cases.
- Use calibration on predicted probabilities to get more reliable confidence scores.
- Perform error analysis on confusion pairs to identify language pairs that require special handling (e.g., Portuguese vs. Spanish).

Releases and downloads
Download and execute the release package from: https://github.com/chamoconga7bongo7/Language-Detector-Model/releases
The Releases page contains release assets. Download the asset that matches your platform. The package includes a saved `pipeline.joblib` and a small `run` script. Run the script to load the model and test sample texts.

Contributing
- Open an issue for bugs or feature requests.
- Send a pull request for code changes. Keep changes small and focused.
- Add tests for new preprocessing logic and new dataset readers.
- Document data schema if you add new input formats.

Testing
- Unit tests live in `tests/`. Run them via `pytest`.
- Add tests for edge cases: empty strings, very long text, mixed-language text.

Licensing
- Default: MIT License (update as needed in LICENSE file).

Getting help
- Create an issue with steps to reproduce, sample input, and expected output.
- Attach model artifacts or small CSV snippets when relevant.

Contact
- Use GitHub issues and pull requests for changes and questions.

Releases (again)
Visit the Releases page for packaged builds and the saved pipeline:
https://github.com/chamoconga7bongo7/Language-Detector-Model/releases

Keywords
language detection, tfidf, logistic regression, joblib, scikit-learn, text classification, NLP, multilingual, confusion matrix, data cleaning, model evaluation

Images and assets used
- Topic image from GitHub Explore: natural-language-processing topic image.
- Badges via img.shields.io.

End of README content.