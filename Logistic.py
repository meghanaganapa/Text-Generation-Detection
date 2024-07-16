import json
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Define a function to load data from JSON files
def load_data(file_paths):
    data = []
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            for line in file:
                json_data = json.loads(line)
                data.append((json_data['text'], json_data['label']))
    return data

# Load data from JSON files
file_paths = ["domain1_train_data.json"]
data = load_data(file_paths)

# Separate text and labels
X = [sample[0] for sample in data]
y = [sample[1] for sample in data]


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using Count Vectorization
vectorizer = CountVectorizer(tokenizer=lambda x: x, lowercase=False)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


# Initialize and train logistic regression model with Lasso regularization
logistic_regression = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
logistic_regression.fit(X_train_vectorized, y)

# Predict on the testing set
y_pred = logistic_regression.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)