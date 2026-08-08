from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from train import train_model


def evaluate_model():
    model, X_test_scaled, y_test, scaler = train_model()
    predictions = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print("Model Evaluation")
    print("----------------")
    print(f"MAE : {mae:.4f}")
    print(f"MSE : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2 : {r2:.4f}")


if __name__ == "__main__":
    evaluate_model()
