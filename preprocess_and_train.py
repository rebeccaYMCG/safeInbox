import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
import re
import string
import joblib


# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    return text


# Load the dataset
file_path = 'data/phishingEmail.csv'
data = pd.read_csv(file_path)

# Preprocess the text data
data['text'] = data['text'].apply(preprocess_text)

# Encode the labels ('spam' and 'real' or 'safe') to numerical values
label_encoder = LabelEncoder()
data['type'] = label_encoder.fit_transform(data['type'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['type'], test_size=0.2, random_state=42)

# Create a pipeline for TF-IDF transformation and model training
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Save the trained model and label encoder
joblib.dump(model, 'model/phishing_detector.pkl')
joblib.dump(label_encoder, 'model/label_encoder.pkl')

print("Model training complete. Model and label encoder saved.")
