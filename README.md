# End-to-End Hybrid Fake News Detection System

This is a realistic final-year project that verifies news in two stages:
1. Official source and portal similarity verification.
2. Machine learning fallback classification.

It now includes a full **verification history module** using Flask + SQLite + frontend history table.

## 1. Problem Statement
Fake news detection should not rely only on text classification. This project combines trusted-source checking, official article comparison, and ML fallback to produce transparent decisions.

## 2. Core Workflow
1. User submits news text and optional source URL.
2. System preprocesses text and extracts keywords/entities.
3. Related official articles are fetched from BBC, Reuters, The Hindu, NDTV via Google News RSS query.
4. Similarity is computed (TF-IDF cosine, optional embeddings).
5. If official evidence is strong: `Real News (Verified Official Source)`.
6. Otherwise ML (TF-IDF + Logistic Regression/Naive Bayes) classifies news.
7. Decision engine outputs final label and explanation.
8. History record is saved in SQLite for every check.

## 3. Technology Stack
- Python
- Flask
- SQLite
- Scikit-learn
- Pandas, NumPy
- NLTK
- Requests, Feedparser
- HTML, CSS, JavaScript

## 4. Project Structure
```text
fak new/
- app.py
- Procfile
- requirements.txt
- README.md
- data/
  - README.md
  - sample_fake_news.csv
  - history.db                          # auto-created
- docs/
  - flow_diagram.md
  - system_architecture.md
  - figures/                            # generated after training
- models/
  - best_model.pkl
  - model_comparison.csv
- src/
  - __init__.py
  - preprocess.py
  - source_verifier.py
  - portal_verifier.py
  - similarity.py
  - decision_engine.py
  - hybrid_service.py
  - history_db.py
  - train.py
- static/
  - style.css
  - script.js
- templates/
  - index.html
```

## 5. Dataset
Preferred: Kaggle Fake and Real News Dataset  
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Place:
- `data/Fake.csv`
- `data/True.csv`

Fallback demo file: `data/sample_fake_news.csv`

## 6. Setup and Run
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.train
python app.py
```
Open `http://127.0.0.1:5000`.

## 6A. GitHub Pages Frontend Deployment
This project includes a static frontend in `docs/` for GitHub Pages.

1. Deploy backend first (Render/Heroku/Railway).
2. Set backend URL in `docs/config.js`:
   - `window.API_BASE_URL = "https://your-backend-url";`
3. In GitHub repo settings:
   - `Settings -> Pages`
   - Source: `Deploy from branch`
   - Branch: `main`
   - Folder: `/docs`
4. Open your GitHub Pages URL.

If you do not set `docs/config.js` to your backend URL, the UI cannot fetch analysis/history data.

## 7. API Endpoints

### `GET /health`
Health check endpoint.

### `POST /analyze`
Main endpoint for hybrid verification.

Request:
```json
{
  "text": "News content",
  "source_url": "https://www.bbc.com/news/..."
}
```

Response includes:
- `result` (`Real`, `Fake`, `Unverified`)
- `verification_method` (`Official Source Comparison` or `Machine Learning`)
- `final_label`
- `reasoning`
- `confidence`
- `similarity`
- `matched_article`
- `keywords`
- `entities`

### `POST /predict`
Alias of `/analyze`.

### `GET /history`
Returns history in reverse chronological order.  
Query params:
- `page`
- `limit`
- `result` filter (`Real`, `Fake`, `Unverified`)

### `DELETE /history`
Clears all history records.

### `GET /history/export`
Exports history to CSV.

## 8. Verification History Feature
Every verification stores:
- short summary of news text
- source URL (optional)
- verification result (`Real/Fake/Unverified`)
- method used (`Official Source Comparison` or `Machine Learning`)
- date/time

Frontend provides:
- dedicated `Check Authenticity History` section
- table columns: Date, News Summary, Source, Result, Method
- color coding:
  - Green = Real
  - Red = Fake
  - Orange = Unverified
- pagination controls
- result filter dropdown
- clear history button
- export CSV button

History refreshes immediately after each new check.

## 9. Decision Logic
1. Trusted URL from whitelist -> verified real.
2. Else compare with official portal articles.
3. If similarity passes threshold -> verified real.
4. Else ML fallback:
   - high fake confidence -> fake
   - moderate real confidence -> real (unverified)
   - otherwise suspicious/unverified

## 10. Flow and Architecture
- Flow diagram: `docs/flow_diagram.md`
- Architecture: `docs/system_architecture.md`

## 11. Limitations
- Official article retrieval depends on internet/API availability.
- Similarity may miss complex paraphrases.
- ML quality depends on dataset quality and retraining frequency.

## 12. Future Enhancements
- Add NewsAPI/GNews key-based integration.
- Use transformer embeddings by default.
- Add multilingual support.
- Add SHAP/LIME explanation dashboard.

## 13. Viva Summary
This project is a hybrid verification engine. It first checks official corroboration, then uses ML when verification is weak, and always returns explainable output plus audit history.

## 14. Resume-Ready Description
- Built a hybrid fake news verification system combining official portal comparison and TF-IDF based ML classification with an explainable decision engine.  
- Developed Flask + SQLite backend with history tracking, filtering, CSV export, and a responsive frontend for transparent verification workflows.
