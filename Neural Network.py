import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.layers import LeakyReLU # type: ignore
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

test=[]
with open("test_data.json","r") as f:
    for l in f:
        j=json.loads(l)
        test.append((j['text'],j["id"]))

X_test=[sample[0] for sample in test]
X_test_id=[sample[1] for sample in test]
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
#X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Vectorize tokenized sequences using CountVectorizer
vectorizer = CountVectorizer(tokenizer=lambda x: x, lowercase=False)
X_train_vec = vectorizer.fit_transform(texts)
X_test_vec = vectorizer.transform(X_test)

def create_model():
    model = Sequential()
    model.add(Dense(units=8, input_shape=(X_train_vec.shape[1],)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Define KerasClassifier
keras_model = KerasClassifier(build_fn=create_model, verbose=2)
early_stopping = EarlyStopping(monitor='val_loss', patience=0, verbose=1)
# Train the model

# Train the model
keras_model.fit(X_train_vec, labels, batch_size=32, epochs=10)

# Make predictions on the test data
predictions = keras_model.predict(X_test_vec)

# Convert predicted probabilities to binary labels
binary_predictions = np.array((predictions > 0.5).astype(int)).flatten()

results_df = pd.DataFrame({'id': X_test_id, 'label': binary_predictions})

# Save DataFrame to a CSV file
results_df.to_csv('nn_withouttraintest.csv', index=False)