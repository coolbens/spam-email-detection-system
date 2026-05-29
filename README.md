# Spam Email Detection System

A production-style **Spam Email Detection System** built with **Pure NLP + Machine Learning** using **Streamlit**.

This project classifies messages into:

- **Spam**
- **Not Spam (Ham)**

It is designed as a recruiter-friendly portfolio project that demonstrates:

- classical NLP preprocessing
- feature engineering with TF-IDF / Bag of Words
- model comparison with Naive Bayes and Logistic Regression
- probability-based predictions
- keyword-based explainability
- batch prediction from uploaded `.txt` files
- clean multi-file Python architecture

---

## Features

- Sidebar dropdown to switch between Naive Bayes and Logistic Regression
- Colorful interactive bulk-insight graphs powered by Plotly


- Clean modern **dark Streamlit UI**
- Single message prediction
- Bulk prediction via **.txt upload**
- Confidence score / probability score
- Top spam keywords
- Top ham keywords
- Important words influencing prediction
- Naive Bayes vs Logistic Regression comparison
- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix

---

## Tech Stack

- **Python**
- **Streamlit**
- **NLTK**
- **scikit-learn**
- **Pandas**
- **NumPy**

---

## Project Structure

```bash
spam_detector/
│── app/
│   ├── app.py
│   ├── ui.py
│
│── nlp/
│   ├── preprocess.py
│   ├── vectorizer.py
│
│── model/
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│
│── data/
│   ├── dataset.csv
│
│── utils/
│   ├── helpers.py
│
│── assets/
│   └── screenshots/
│
│── requirements.txt
│── README.md
```

---

## NLP Pipeline

The application uses a full classical NLP pipeline:

1. **Lowercasing**
2. **Punctuation removal**
3. **URL / email / number normalization**
4. **Tokenization**
5. **Stopword removal** with NLTK
6. **Lemmatization**
7. **Vectorization**
   - TF-IDF (default)
   - Bag of Words (optional)
8. **Machine learning classification**

### Example preprocessing flow

```text
Original:
"Congratulations! You won a FREE trip. Click now!!!"

Processed:
"congratulation free trip click"
```

---

## Machine Learning Models

The training script compares:

- **Multinomial Naive Bayes**
- **Logistic Regression**

The system automatically selects the model with the best **F1 score** and saves it for inference.

---

## Dataset

This project includes a small demo dataset in:

```bash
data/dataset.csv
```

### Recommended real dataset for stronger results

Use the **SMS Spam Collection Dataset** for better performance in your portfolio demo.

You can replace `data/dataset.csv` with a larger real dataset using the same format:

```csv
label,text
spam,"Win a free phone now"
ham,"Let's meet at 5 PM"
```

Recommended labels:

- `spam`
- `ham`

---

## Model Evaluation

The project reports:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

The best model summary is automatically saved to:

```bash
artifacts/metrics.json
```

---

## UI Screenshots

Sample mock screenshots are included in:

```bash
assets/screenshots/
```

Suggested screenshots for your GitHub repo:

1. Home screen
2. Single message prediction
3. Bulk prediction result
4. Model insights tab

---

## How to Run Locally

### 1. Clone or extract the project

```bash
cd spam_detector
```

### 2. Create a virtual environment

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the models

```bash
python -m model.train
```

### 5. Run the Streamlit app

```bash
streamlit run app/app.py
```

---

## Example Use Cases

### Single message
Paste text like:

```text
Urgent! Your account has been selected for a reward. Reply now.
```

The app will return:

- Spam / Not Spam
- Confidence score
- Probability breakdown
- Important keywords

### Bulk prediction
Upload a `.txt` file with one message per line:

```text
Hi, are we still meeting tomorrow?
You won a free shopping voucher. Click now.
Please review the attached report.
```

The app will return a prediction table and allow CSV export.

---

## Why This Project Is Good for a Portfolio

This project demonstrates real-world AI engineering skills:

- end-to-end NLP pipeline design
- clean multi-file code organization
- ML model comparison
- production-style inference flow
- explainability with keyword influence
- deployable Streamlit user interface

It shows that you understand how to build practical AI systems **without depending on LLM APIs or pretrained transformers**.

---

## Improvement Ideas

To make this even stronger for a portfolio:

- add SVM comparison
- use the full SMS Spam Collection dataset
- add charts for class distribution
- save prediction logs
- add Docker support
- deploy to Streamlit Community Cloud or Render

---

## License

Use freely for portfolio and learning purposes.


## Latest UI Update
- Single prediction now keeps model insights at zero.
- Bulk prediction now accepts Excel files (.xlsx, .xls).
- Model Insights tab only activates after a bulk upload and includes graphs for label counts, average probabilities, confidence trends, and top keywords.


## Production deployment

### Local launch
```bash
pip install -r requirements.txt
streamlit run app/app.py
```

### Safer launcher
```bash
python run.py
```

### Docker
```bash
docker build -t spam-detector .
docker run -p 8501:8501 spam-detector
```

### Docker Compose
```bash
docker compose up --build
```

### Environment and deployment files included
- `Dockerfile`
- `docker-compose.yml`
- `.streamlit/config.toml`
- `Procfile`
- `pyproject.toml`
- `run.py`
- `scripts/healthcheck.py`
- `tests/smoke_test.py`

### Why this version is deployment-ready
- safer import bootstrapping for `streamlit run app/app.py`
- pinned dependency ranges
- automatic NLTK stopwords setup with offline fallback
- container health check
- cloud-friendly `PORT` support
- reproducible startup commands
- simple smoke test for prediction flow
