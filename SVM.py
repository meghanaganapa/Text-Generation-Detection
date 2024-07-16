import json
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

# Read the data
data1 = pd.read_json('domain1_train_data.json', lines=True)
data2 = pd.read_json('domain2_train_data.json', lines=True)

# Perform under-sampling
# Perform under-sampling
under = RandomUnderSampler(sampling_strategy=1, random_state=42)
X, y = under.fit_resample(data2[['text']], data2[['label']])
data_up = pd.concat([X, y], axis=1)
combined_data = pd.concat([data1, data_up[['text', 'label']]], ignore_index=True)

# Split data into training and testing sets
#X_train, X_val, y_train, y_val = train_test_split(combined_data["text"], combined_data["label"], test_size=0.2, random_state=42)

# Define hyperparameters for SVM
svm_params = {
    'C': 10.0,
    'gamma': 'scale',
    'kernel': 'rbf'
}

# Create pipeline with vectorizer and classifier
svm_pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=lambda x: x, lowercase=False)),  # Tokenizer set to None
    ('clf', SVC(**svm_params))  # Support Vector Machine classifier with specified parameters
])

vectorizer = CountVectorizer(tokenizer=lambda x: x, lowercase=False)
X_train_vec = vectorizer.fit_transform(combined_data["text"])

# Fit the pipeline on the training data
svm_pipeline.fit(combined_data["text"], combined_data["label"])
test=[]
with open("test_data.json","r") as f:
    for l in f:
        j=json.loads(l)
        test.append((j['text'],j["id"]))

X_test=[sample[0] for sample in test]
X_test_id=[sample[1] for sample in test]
X_val_vec = vectorizer.transform(X_test)
y_pred=svm_pipeline.predict(X_test)

results_df = pd.DataFrame({'id': X_test_id, 'label': y_pred})

# Save DataFrame to a CSV file
results_df.to_csv('svm_combined_withouttraintest.csv', index=False)
# Get confidence scores for the positive class (class 1)

