import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting application...")

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Function to preprocess text
def preprocess_text(text, tokenizer, max_length=128):
    encoded_input = tokenizer(text, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    return encoded_input['input_ids'], encoded_input['attention_mask']

# Function to get BERT embeddings
def get_bert_embeddings(text_list, tokenizer, bert_model):
    embeddings = []
    with torch.no_grad():
        for text in text_list:
            input_ids, attention_mask = preprocess_text(text, tokenizer)
            outputs = bert_model(input_ids, attention_mask=attention_mask)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embedding.flatten())
    return np.array(embeddings)

# Sample dataset
reviews = [
    "I love this product!",
    "This is the worst experience I've ever had.",
    "It's okay, not great.",
    "Absolutely fantastic!",
    "Terrible, would not recommend.",
    "I hate this item.",
    "This is amazing!",
    "Not bad, but could be better.",
    "I'm very satisfied.",
    "Awful experience."
]

labels = [5, 1, 3, 5, 1, 1, 5, 3, 5, 1]

# Load pre-trained BERT model and tokenizer
try:
    logger.info("Loading BERT model and tokenizer...")
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    bert_model = BertModel.from_pretrained(model_name)
    logger.info("BERT model and tokenizer loaded successfully")
except Exception as e:
    logger.error(f"Error loading BERT model: {str(e)}")
    raise

# Get BERT embeddings for the reviews
X = get_bert_embeddings(reviews, tokenizer, bert_model)
y = np.array(labels)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression classifier
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test)

# Print evaluation metrics
logger.info("Initial Evaluation Metrics:")
logger.info(f"Accuracy: {accuracy_score(y_test, y_pred)}")
logger.info(f"Classification Report:\n {classification_report(y_test, y_pred, zero_division=1)}")

# Pydantic model for the request body
class ReviewRequest(BaseModel):
    review: str

# Function to predict sentiment and rating
def predict_sentiment_and_rating(review, tokenizer, bert_model, classifier):
    input_embedding = get_bert_embeddings([review], tokenizer, bert_model)
    predicted_label = classifier.predict(input_embedding)[0]

    if predicted_label in [1, 2]:
        sentiment = 'negative'
    elif predicted_label == 3:
        sentiment = 'neutral'
    else:
        sentiment = 'positive'

    return sentiment, int(predicted_label)

# Endpoint to handle both GET and POST requests
@app.post("/predict")
@app.get("/predict")
def get_prediction(request: ReviewRequest):
    sentiment, rating = predict_sentiment_and_rating(request.review, tokenizer, bert_model, classifier)
    return {"review": request.review, "sentiment": sentiment, "rating": rating}

# Get port from environment variable
port = int(os.getenv("PORT", 8080))

# Note: Remove the if __name__ == "__main__": block