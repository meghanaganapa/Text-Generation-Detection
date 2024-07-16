import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Read the data
data1 = pd.read_json('domain1_train_data.json', lines=True)
data2 = pd.read_json('domain2_train_data.json', lines=True)

# Perform under-sampling for domain 2
under = RandomUnderSampler(sampling_strategy=1, random_state=42)
X, y = under.fit_resample(data2[['text']], data2[['label']])
data2_resampled = pd.concat([X, y], axis=1)

# Combine data1 and domain2_resampled
combined_data = pd.concat([data1, data2_resampled])

# Shuffle the combined data
combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Check the shape of the combined data
print("Shape of combined data:", combined_data.shape)

# Separate text and labels
texts = np.array(combined_data['text'])
labels = np.array(combined_data['label'])

# Split data into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Vectorize tokenized sequences using CountVectorizer
vectorizer = CountVectorizer(tokenizer=lambda x: x, lowercase=False)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# Initialize XGBoost Classifier with best parameters
best_params = {'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 250}
model = XGBClassifier(**best_params)

# Train the model
model.fit(X_train_vec, y_train)

# Make predictions
y_pred = model.predict(X_val_vec)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
print("Accuracy:", accuracy)
test=[]
with open("test_data.json","r") as f:
    for l in f:
        j=json.loads(l)
        test.append((j['text'],j["id"]))

X_test=[sample[0] for sample in test]
X_test_id=[sample[1] for sample in test]
X_val_vec = vectorizer.transform(X_test)
y_pred=model.predict(X_val_vec)
results_df = pd.DataFrame({'id': X_test_id, 'label': y_pred})

# Save DataFrame to a CSV file
results_df.to_csv('xgboost_combined.csv', index=False)