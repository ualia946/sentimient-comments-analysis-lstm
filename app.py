"""
app.py

This is the main Flask application for the Sentiment Analysis Chatbot.

The app serves a simple web interface where users can enter a comment.
The comment is processed using a trained deep learning model to predict
its sentiment as "Positivo", "Negativo", or "Neutro".

Modules:
- index(): Renders the main input form.
- analizar(): Handles the form submission, runs the prediction, and returns the result.
"""

from flask import Flask, render_template, request
from modelo.predictor import predecir_sentimiento

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analizar", methods=["POST"])
def analizar():
    """
    POST route to receive the comment, process it, and return the result.
    """
    comentario = request.form["comentario"]
    resultado = predecir_sentimiento(comentario) 
    return render_template("index.html", resultado=resultado, comentario=comentario)


if __name__ == "__main__":
    app.run(debug=True)