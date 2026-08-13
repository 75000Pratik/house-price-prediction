# House Price Prediction API

A production-ready Machine Learning API that predicts California house prices using a tuned Random Forest model.

The project covers the complete ML lifecycle: data preprocessing, model training, model evaluation, API development, Dockerization, automated testing, CI/CD, deployment, rollback, and production observability.

## Model Performance

The final tuned Random Forest model achieved approximately:

- MAE: 0.3280
- RMSE: 0.5061
- R2: 0.8045

The model was selected after comparing Linear Regression, Decision Tree, and Random Forest models.

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- Flask
- Gunicorn
- joblib
- Flasgger / Swagger
- pytest
- Docker
- GitHub Actions
- Git LFS
- Docker Hub
- Render

## Project Features

- Random Forest house-price prediction
- Input validation for missing, unexpected, and non-numeric features
- REST API with `/health`, `/version`, and `/predict`
- Swagger API documentation
- Automated pytest test suite
- Dockerized Flask API served with Gunicorn
- Git LFS for large ML model artifacts
- GitHub Actions CI/CD pipeline
- Automatic Docker Hub image publishing
- Docker images tagged with both `latest` and Git commit SHA
- Automatic Render deployment
- Production build-SHA verification
- Production prediction smoke testing
- Rollback using immutable Docker SHA tags
- Prediction latency logging
- Request ID logging and client-visible request IDs

## API Endpoints

### Health Check

```text
GET /health
```
