# 💬 Sentiment Analysis with LSTM

This project applies **Deep Learning** techniques to classify text comments into **positive**, **neutral**, or **negative** sentiments using **LSTM (Long Short-Term Memory)** layers.

The model is trained using a custom dataset generated with ChatGPT and stored in CSV files. After training, it is integrated into a **Flask web app** that allows users to input text and get sentiment predictions in real time.

---

## 🧠 Tech Stack

- Python
- TensorFlow / Keras
- Flask (Web App)
- NLTK for preprocessing
- CSV-based custom dataset
- HTML / Jinja for frontend templates

---

## 🧪 Dataset

The dataset consists of synthetic comments created using ChatGPT. Each entry is labeled as positive, neutral, or negative and saved in CSV files.

Please note that the dataset contains **Spanish-language comments**, as this project was developed for a university assignment in Spain.

If you would like to use the chatbot in a different language, you will need to prepare a new dataset in that language, following the same structure used in this project.


Before training:
- The text is cleaned (lowercased, punctuation removed, stopwords removed).
- Tokenization and padding are applied.
- Text is converted into sequences using Keras' `Tokenizer`.

---

## 🛠️ How to Run the Project

### 🔬 Train the Model

1. Open the notebook `model.ipynb`.
2. Run all cells to preprocess data, train the LSTM model, and save it.

### 🌐 Run the Web App

```bash
pip install -r requirements.txt
python app.py

## 📂 Project Structure

Below is the folder structure of the project:
sentiment-analysis-lstm/ ├── app.py # Flask web server ├── modelo/ │ └── predictor.py # Loads model and makes predictions ├── model.ipynb # Notebook with model training ├── modelChatBot.h5 # Trained model ├── modeltokenizer.pickle # Tokenizer for text preprocessing ├── static/ # CSS or images if needed ├── templates/ │ └── index.html # Web frontend ├── data/ # CSV files with comments │ └── Comentarios1.csv │ └── Comentarios2.csv ├── requirements.txt # All dependencies └── .gitignore # Ignore unnecessary files
