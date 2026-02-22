# Spam Classification (Hybrid Feature Selection + Deep Learning)

## Setup (Windows)
1. Create & activate venv
   - `python -m venv venv`
   - `venv\Scripts\Activate.ps1`

2. Install dependencies
   - `pip install -r requirements.txt`

## Train Model
From project root:
- `python -m backend.model.train`

Saved outputs:
- `backend/saved_model/hybrid_fs.joblib`
- `backend/saved_model/spam_nn.keras`

## Run API
From project root:
- `python -m backend.app`

Open:
- http://127.0.0.1:5000/

## Run Frontend
Open:
- `frontend/index.html`