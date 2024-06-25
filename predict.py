from model import preprocess_text, load_model_and_encoder

# Load the trained model and label encoder
model, label_encoder = load_model_and_encoder()


# Function to predict if an email is spam or not
def predict_email(email_content):
    preprocessed_text = preprocess_text(email_content)
    prediction_result = model.predict([preprocessed_text])
    label = label_encoder.inverse_transform(prediction_result)
    return label[0]


# Example usage
email_text = "Your account has been compromised. Please click the link to reset your password."
prediction = predict_email(email_text)
print(f'The email is predicted to be: {prediction}')
