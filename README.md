# House Price Prediction API

A machine learning API that predicts house prices using a trained Random Forest model.

## Tech Stack

- Python
- Flask
- scikit-learn
- joblib

## Run locally

1. Clone the repository:

```bash
git clone https://github.com/75000Pratik/house-price-prediction.git
cd house-price-prediction
```

2. Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start the API:

```bash
python app/app.py
```

The API runs at:

```text
http://127.0.0.1:5000
```

## Health Check

```text
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "model": "Random Forest"
}
```

## Prediction

```text
POST /predict
Content-Type: application/json
```
