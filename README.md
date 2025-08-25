# 📰 Fake News Classification API

Welcome to the **Fake News Classification API**, a powerful tool designed to detect fake news articles using advanced machine learning techniques. Built with **FastAPI** and powered by **XGBoost** and **RandomForest** models, this project aims to combat misinformation by classifying news articles as "Fake" or "Not Fake" based on their text content. 🚀

---

## 🌟 Project Overview

### What We Did
This project develops a machine learning-based API to classify news articles as fake or not fake. The pipeline includes:
- **Data Preprocessing**: Using **spaCy** to clean and prepare text data by:
  - Converting text to lowercase.
  - Removing stopwords to eliminate common words (e.g., "the", "is").
  - Applying lemmatization to reduce words to their base forms (e.g., "running" → "run").
- **Feature Extraction**: Utilizing `TfidfVectorizer` with bigrams (`ngram_range=(1,2)`) and a maximum of 10,000 features to convert text into numerical representations.
- **Model Training**: Two baseline models were trained:
  - **RandomForest**: Achieved an impressive **99.95% accuracy** on the test set, with a confusion matrix showing 1 false positive and 0 false negatives.
  - **XGBoost**: Achieved **100% accuracy** with a perfect confusion matrix, making it the final model saved as `fake_news_classification_model.pkl`.
- **API Development**: A **FastAPI** application serves predictions via a REST endpoint (`/predict/`), accepting text input and returning a classification ("Fake" if fake probability > 90%, else "Not Fake") along with probability scores.
- **Deployment Readiness**: The API is designed for easy deployment, e.g., on an EC2 instance, with Uvicorn and Gunicorn for production.

### Use of the Project
The Fake News Classification API enables users to:
- **Classify News Articles**: Input a news article’s text to determine if it’s fake or not, helping users verify information in real-time.
- **Integrate with Applications**: The REST API can be integrated into web or mobile apps, news aggregators, or content moderation systems.
- **Combat Misinformation**: Provides a scalable solution for platforms to filter out fake news, enhancing content credibility.

### Importance
In today’s digital age, misinformation spreads rapidly, influencing public opinion, elections, and societal trust. This project addresses:
- **Trustworthy Information**: Helps users and platforms identify reliable news sources.
- **Scalable Detection**: Offers an automated, high-accuracy solution to detect fake news at scale.
- **Real-World Impact**: Supports journalists, researchers, and social media platforms in combating fake news, fostering a more informed society.

---

## 🛠️ Technical Details

### Dataset
- **Source**: The dataset consists of two CSV files:
  - `Fake.csv`: 23,481 fake news articles.
  - `True.csv`: 21,417 true news articles.
- **Features**: Each article includes `title`, `text`, `subject`, and `date`. Only the `text` column is used for classification after preprocessing.
- **Labels**: Fake articles are labeled as `0`, true articles as `1`.

### Preprocessing
Text data is preprocessed using **spaCy**:
- **Lowercasing**: Converts text to lowercase for consistency.
- **Stopword Removal**: Removes common English stopwords to focus on meaningful words.
- **Lemmatization**: Reduces words to their base forms (e.g., "writes" → "write").
- **Token Filtering**: Keeps only alphabetic tokens to exclude numbers and punctuation.

### Model Pipeline
The classification pipeline, implemented with **scikit-learn**, includes:
- **TfidfVectorizer**:
  - Parameters: `ngram_range=(1,2)`, `max_features=10000`.
  - Converts preprocessed text into TF-IDF features.
- **Baseline Models**:
  - **RandomForest**: Trained with default parameters, achieving 99.95% accuracy.
  - **XGBoost**: Configured with `n_estimators=300`, `learning_rate=0.1`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`, `eval_metric="logloss"`, achieving 100% accuracy.
- **Saved Model**: The XGBoost model is saved as `fake_news_classification_model.pkl` for its superior performance.

### API Functionality
- **Endpoint**: `POST /predict/`
  - Accepts JSON input: `{"text": "Your news article text"}`.
  - Returns: `{"prediction": "Fake" or "Not Fake", "probability": {"Fake": float, "Not Fake": float}}`.
  - Prediction Logic: Classifies as "Fake" if the fake class probability exceeds 90%, otherwise "Not Fake".
- **Root Endpoint**: `GET /` returns a welcome message.

### Performance
- **RandomForest**: 99.95% accuracy, near-perfect confusion matrix.
- **XGBoost**: 100% accuracy, perfect confusion matrix on a 2,000-sample test set.
- **Note**: Perfect accuracy suggests potential overfitting; real-world performance should be monitored.

---

## 🚀 Getting Started

### Prerequisites
- **Python**: 3.8 or higher.
- **Dependencies**:
  ```bash
  pip install fastapi uvicorn scikit-learn==1.6.1 xgboost spacy
  python -m spacy download en_core_web_sm
  ```
- **Model File**: Ensure `fake_news_classification_model.pkl` is in the project directory.

### Installation
1. Clone the repository or create a project directory:
   ```bash
   mkdir Fake_News_Classification
   cd Fake_News_Classification
   ```
2. Save the FastAPI code as `app.py` (provided in the project).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Or manually:
   ```bash
   pip install fastapi uvicorn scikit-learn==1.6.1 xgboost spacy
   python -m spacy download en_core_web_sm
   ```
4. Ensure `fake_news_classification_model.pkl` is in the directory.

### Running the API Locally
1. Activate your virtual environment (e.g., `MTNM`):
   ```bash
   .\MTNM\Scripts\activate  # Windows
   source MTNM/bin/activate  # Linux/Mac
   ```
2. Run the FastAPI server:
   ```bash
   python app.py
   ```
   Or directly with Uvicorn:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```
3. Access the API:
   - **Swagger UI**: `http://localhost:8000/docs`
   - **Root**: `http://localhost:8000/`
   - **Predict**: `POST http://localhost:8000/predict/`

### Example Request
```json
{
    "text": "This is a sample news article text for classification."
}
```
**Response** (example):
```json
{
    "prediction": "Not Fake",
    "probability": {
        "Fake": 0.12,
        "Not Fake": 0.88
    }
}
```

---

## 🌍 Deployment

### Local Deployment
Use Gunicorn with Uvicorn for production-like local testing:
```bash
pip install gunicorn
gunicorn -k uvicorn.workers.UvicornWorker app:app -b 0.0.0.0:8000
```

### Cloud Deployment (e.g., AWS EC2)
1. Set up an EC2 instance with Ubuntu or Amazon Linux.
2. Install dependencies:
   ```bash
   sudo apt update
   sudo apt install python3-pip
   pip3 install fastapi uvicorn scikit-learn==1.6.1 xgboost spacy gunicorn
   python3 -m spacy download en_core_web_sm
   ```
3. Copy `app.py` and `fake_news_classification_model.pkl` to the instance.
4. Run the server:
   ```bash
   gunicorn -k uvicorn.workers.UvicornWorker app:app -b 0.0.0.0:8000
   ```
5. Configure security groups to allow port 8000.

---

## 🛠️ Troubleshooting

- **scikit-learn Version Mismatch**:
  - If you see `InconsistentVersionWarning`, ensure scikit-learn 1.6.1 is installed:
    ```bash
    pip uninstall scikit-learn
    pip install scikit-learn==1.6.1
    ```
  - Alternatively, retrain the model with scikit-learn 1.7.1 using the original training script.
- **Silent Termination**:
  - Check for errors with:
    ```bash
    python -v app.py
    ```
  - Ensure `fake_news_classification_model.pkl` is in the correct directory.
- **spaCy Model Issues**:
  - Verify `en_core_web_sm` is installed:
    ```bash
    python -m spacy validate
    ```

---

## 🎯 Future Improvements
- **Real-World Testing**: Validate model performance on diverse, unseen datasets to address potential overfitting.
- **Additional Preprocessing**: Incorporate URL removal or entity recognition if needed.
- **Scalability**: Implement batch processing for high-throughput requests.
- **Monitoring**: Add logging to track API usage and prediction performance.

---

## 📚 References
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [spaCy](https://spacy.io/)
- [Uvicorn](https://www.uvicorn.org/)
- [Gunicorn](https://gunicorn.org/)

---

## 🙌 Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request with detailed changes.

---

## 📜 License
This project is licensed under the MIT License.

---

**Built with ❤️ by the Fake News Classification Team**