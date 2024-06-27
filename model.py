import re
import string
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    return text


# Function to load the trained model and label encoder
def load_model_and_encoder():
    model = joblib.load('model/phishing_detector.pkl')
    label_encoder = joblib.load('model/label_encoder.pkl')
    return model, label_encoder


# Function to create and return a new model pipeline
def create_model_pipeline():
    return make_pipeline(TfidfVectorizer(), MultinomialNB())


# Function to train the model with given data
def train_model(X_train, y_train):
    model = create_model_pipeline()
    model.fit(X_train, y_train)
    return model


# Function to save the model and label encoder
def save_model_and_encoder(model, label_encoder):
    joblib.dump(model, 'model/phishing_detector.pkl')
    joblib.dump(label_encoder, 'model/label_encoder.pkl')
