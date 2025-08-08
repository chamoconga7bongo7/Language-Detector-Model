import time
import os
import pandas as pd
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer , CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline , make_pipeline
from Cleaner_DS import dataset_Analyse
import joblib
from sklearn.feature_extraction.text import HashingVectorizer
import traceback
import numpy as np

matplotlib.use("TkAgg")

start_time = datetime.now()
print("Started Train Time :",start_time)


analyzer = dataset_Analyse()
main = analyzer.Dataset()
cleaned = analyzer.Cleaned_DS()

counts = cleaned['language_full'].value_counts()
to_keep = counts[counts >= 5].index
cleaned = cleaned[cleaned['language_full'].isin(to_keep)]

class Train_Model:

    def Main(self):
        path_for_model = "Model"
        try:
            if os.path.exists(path_for_model):
                print(f"Directory :{path_for_model} already exist ...")
            else:
                os.makedirs(path_for_model,exist_ok=True)


            data = cleaned
            x = data["text"]
            y = data["language_full"]

            x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.3,random_state=42,stratify=y)
            pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(analyzer='char', max_features=2500)),
                ("clf", LogisticRegression(max_iter=2500, solver='saga'))
            ])
            pipeline.fit(x_train,y_train)

            y_pred = pipeline.predict(x_test)
            print("Accuracy Score :",accuracy_score(y_test,y_pred))
            print("Classification Report :",classification_report(y_test,y_pred))

            cmap = sns.color_palette("tab20c", as_cmap=True)
            confusion = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
            plt.figure(figsize=(25, 22))
            sns.heatmap(confusion, fmt='d', annot=True, cmap=cmap,
                        xticklabels=pipeline.classes_,
                        yticklabels=pipeline.classes_,
                        linewidths=0.5, linecolor='gray', cbar=True,
                        annot_kws={"size": 6})

            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title("Confusion Matrix")
            plt.xticks(rotation=90, fontsize=8)
            plt.yticks(fontsize=8)
            plt.tight_layout()
            plt.savefig("Confusion_Matrix.png")
            plt.close()

            true_positives = np.diag(confusion)
            classes = pipeline.classes_

            plt.figure(figsize=(25, 12))
            plt.plot(classes, true_positives, marker='o', linestyle='-', color='b')
            plt.xticks(rotation=90, fontsize=7)
            plt.xlabel("Classes", fontsize=14)
            plt.ylabel("Number of Correct Predictions (True Positives)", fontsize=14)
            plt.title("True Positives per Class (Line Plot)", fontsize=16)
            plt.tight_layout()
            plt.savefig("True_Positives_Line_Plot.png")
            plt.close()

            plt.figure(figsize=(25, 8))
            plt.bar(classes, true_positives, color='orange')
            plt.xticks(rotation=90, fontsize=7)
            plt.xlabel("Classes", fontsize=14)
            plt.ylabel("Number of Correct Predictions (True Positives)", fontsize=14)
            plt.title("True Positives per Class (Histogram)", fontsize=16)
            plt.tight_layout()
            plt.savefig("True_Positives_Histogram.png")
            plt.close()

            try:
                joblib.dump(pipeline, "Model/model.pkl")
                print("Succesfully Saved Model :",pipeline)
            except Exception as e:
                print(f"Failed to Save model : {e}")

        except Exception as e:
            print("Failed to Train Model:")
            traceback.print_exc()


C = Train_Model()
C.Main()

end_time = datetime.now()
print("Ended Time :",end_time)
