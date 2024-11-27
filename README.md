
# Sentiment Analysis 

## Overview
This project provides a FastAPI-based sentiment analysis API that leverages a pre-trained BERT model and Logistic Regression to predict the sentiment and rating of text reviews.

## Features
- **API Framework**: Built using FastAPI with endpoints for sentiment prediction.
- **Deep Learning**: Utilizes BERT embeddings from the Hugging Face Transformers library.
- **Machine Learning**: Logistic Regression for classification.
- **Docker Support**: Ready for containerized deployment.
- **Testing**: Includes a `test.py` script to validate API functionality.

## File Structure
- `main.py`: Main application file defining the FastAPI server and sentiment prediction logic.
- `test.py`: Script to test the API using sample reviews.
- `requirements.txt`: List of dependencies.
- `Dockerfile`: Configuration for building and running the project as a Docker container.

## Installation

### Prerequisites
- Python 3.8 or higher.
- Docker (for containerized deployment).

### Steps
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the FastAPI server:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8080 --reload
   ```

4. Access API documentation at [http://127.0.0.1:8080/docs](http://127.0.0.1:8080/docs).

## Usage

### Predict Sentiment
Use the `/predict` endpoint with a POST or GET request. Example request body:
```json
{
  "review": "This product exceeded my expectations!"
}
```

Response:
```json
{
  "review": "This product exceeded my expectations!",
  "sentiment": "positive",
  "rating": 5
}
```

### Test API
Run the `test.py` script to test the API:
```bash
python test.py
```

## Docker Deployment

1. Build the Docker image:
   ```bash
   docker build -t sentiment-analysis-api .
   ```

2. Run the container:
   ```bash
   docker run -p 8080:8080 sentiment-analysis-api
   ```

3. Access the API at [http://127.0.0.1:8080](http://127.0.0.1:8080).

## Dependencies
- `fastapi`: Framework for building APIs.
- `uvicorn`: ASGI server for FastAPI.
- `pydantic`: Data validation.
- `torch`: Deep learning library.
- `transformers`: Hugging Face library for NLP.
- `scikit-learn`: Machine learning utilities.
- `numpy`: Numerical computations.

