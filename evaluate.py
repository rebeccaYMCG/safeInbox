import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from model import preprocess_text, load_model_and_encoder

# Load the trained model and label encoder
model, label_encoder = load_model_and_encoder()

# Load new data for evaluation
file_path = 'Downloads\new_phishingData.csv'
data = pd.read_csv(file_path)

# Preprocess the text data (same preprocessing as training)
data['text'] = data['text'].apply(preprocess_text)
data['type'] = label_encoder.transform(data['type'])

# Evaluate the model
X_test = data['text']
y_test = data['type']
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
