import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from data_loader import load_data
from preprocess import preprocess_data


def train_model():
    df = load_data()

    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(
        df)

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/best_random_forest.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    return model, X_test_scaled, y_test, scaler


if __name__ == "__main__":
    model, X_test_scaled, y_test, scaler = train_model()
    print("model trained successfully")
    print("Test data shape:", X_test_scaled.shape)
