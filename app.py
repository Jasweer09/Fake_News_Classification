import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import spacy
import uvicorn

# Debugging: Print Python and package versions
print("Python version:", sys.version)
print("Loading FastAPI app...")

# Initialize FastAPI app
app = FastAPI(title="Fake News Classification API")

# Define input schema using Pydantic
class NewsInput(BaseModel):
    text: str  # Only text field is required

# Load spaCy model
try:
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])  # Disable unused components
    print("spaCy model loaded successfully")
except Exception as e:
    print(f"Failed to load spaCy model: {str(e)}")
    raise

# Load the saved model
try:
    print("Loading pickled model...")
    with open("fake_news_classification_model.pkl", "rb") as f:
        classifier = pickle.load(f)
    print("Model loaded successfully")
except Exception as e:
    print(f"Failed to load model: {str(e)}")
    raise

def preprocess_text(text: str) -> str:
    """
    Preprocess the input text using spaCy to match training pipeline:
    - Convert to lowercase
    - Remove stopwords
    - Apply lemmatization
    """
    print("Preprocessing text...")
    # Strip whitespace
    text = text.strip()
    
    # Validate input
    if not text:
        raise ValueError("Input text cannot be empty")
    
    # Process with spaCy (lowercase applied)
    doc = nlp(text.lower())
    
    # Remove stopwords, lemmatize, keep alphabetic tokens
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    
    # Join tokens into a single string
    processed_text = ' '.join(tokens)
    print("Text preprocessing complete")
    return processed_text

@app.post("/predict/")
async def predict(news: NewsInput):
    try:
        # Preprocess the input text
        processed_text = preprocess_text(news.text)
        
        # Transform and get probabilities
        print("Making prediction...")
        prediction_proba = classifier.predict_proba([processed_text])[0]
        print("Prediction complete")

        # Determine label based on fake class probability (>90% for Fake)
        fake_proba = prediction_proba[1]  # Fake is label 1
        label = "Fake" if fake_proba > 0.9 else "Not Fake"

        return {
            "prediction": label,
            "probability": {
                "Fake": float(prediction_proba[1]),
                "Not Fake": float(prediction_proba[0])
            }
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to the Fake News Classification API. Use POST /predict/ to classify news."}

if __name__ == "__main__":
    print("Starting Uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)