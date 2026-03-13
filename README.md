# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import dataset and preprocess the text data (clean and convert messages into numerical form using TF-IDF). 
2.Split the dataset into training and testing sets. 
3.Train the Support Vector Machine (SVM) model using the training data from Machine Learning.
4.Test the model on unseen data and evaluate performance using a confusion matrix and accuracy score 
```
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Anikha Pillai
RegisterNumber: 25009524 
*/

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')

# Keep only useful columns
df = df[['v1','v2']]
df.columns = ['label','message']

df.head()

# Convert spam/ham to numeric
df['label'] = df['label'].map({'ham':0, 'spam':1})

df.head()

# Convert text to numerical data
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['message'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SVC(kernel='linear')

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham','Spam'],
            yticklabels=['Ham','Spam'])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - SVM Spam Detection")
plt.show()

print(classification_report(y_test, y_pred))
```

## Output:

<img width="665" height="509" alt="Screenshot 2026-03-13 112707" src="https://github.com/user-attachments/assets/ffd90faa-1e29-4f7b-ae0f-e73a9ce60ca6" />
<img width="558" height="187" alt="Screenshot 2026-03-13 112726" src="https://github.com/user-attachments/assets/6cfe1b52-9a68-4d9b-b198-829f2b9c4ea8" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
