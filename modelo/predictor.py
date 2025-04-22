"""
predictor.py

This module is responsible for loading the trained sentiment analysis model and tokenizer,
and using them to classify the sentiment of user-provided comments in the chatbot interface.

The model was trained on text comments labeled as "positive", "neutral", or "negative",
and this script provides the backend logic for real-time predictions.

Steps:
- Load the trained tokenizer and model.
- Preprocess the input comment (tokenization and padding).
- Predict the sentiment class.
- Return a readable label: "Positivo" (Positive), "Neutro" (Neutral), or "Negativo" (Negative).
"""

# Load necessary libraries and custom components
import pickle
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Bidirectional
import numpy as np

# Load the tokenizer that was used during model training
with open("modelo/modeltokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

# If the model contains custom layers like Bidirectional or LSTM, we register them here
custom_objects = {
    "LSTM": LSTM,
    "Bidirectional": Bidirectional,
}

# Load the trained model
model = keras.models.load_model("modelo/modelChatBot.h5")

# Define the maximum sequence length used during training
max_length = 50

def predecir_sentimiento(comentario):
    """
    Predict the sentiment of a user-provided comment.

    Parameters:
    comentario (str): The input comment from the user.

    Returns:
    str: One of "Negativo", "Neutro", or "Positivo" depending on the predicted class.
    """

    # Convert the comment to a tokenized sequence
    sequence = tokenizer.texts_to_sequences([comentario])
    print("Secuencia tokenizada:", sequence)

    # Apply padding to match the input shape expected by the model
    padded = pad_sequences(sequence, maxlen=max_length, padding="post", truncating="post")
    print("Secuencia padded:", padded)

    # Predict the sentiment using the trained model
    prediccion = model.predict(padded)
    categoria = np.argmax(prediccion)

    # Map prediction to sentiment label
    if categoria == 0:
        return "Negativo"
    elif categoria == 1:
        return "Neutro"
    elif categoria == 2:
        return "Positivo"
