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
email_text = "a resume john , this is a resume i received today from my friend . please , take a look at it . what follows below is a copy of his message to me : dear vincent , i very much would like to ask you for a career advice . i am looking for new challenges and new professional opportunities . possibly there would be such opportunity around yourself at enron corporation . i trust that my strongest asset is my intellectual capital and ability to look from new angles into complex issues . beside of the experience of working under jacob goldfield an paul jacobson at goldman on the interest rate swaps and proprietary desks , i was a part of research effort of john meriwether group at salomon brothers , i headed the european interest options desk at dkbi in london and i have managed a small hedge fund in partnership with albert friedberg . i hold ph . d . in mathematics from mit and i have studied under nobel laureate in economics , bob merton . i very much would like to apply my knowledge of capital markets , trading and research in the field of energy markets . with my very best regards and personal wishes"
prediction = predict_email(email_text)
print(f'The email is predicted to be: {prediction}')
