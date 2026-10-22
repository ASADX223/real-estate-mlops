
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# 1. Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "housing.csv"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "price_model.pkl"
DATABASE_PATH = PROJECT_DIR / "mlflow.db"

# 2. Configure MLflow with SQLite
mlflow.set_tracking_uri(f"sqlite:///{DATABASE_PATH}")
mlflow.set_experiment("real-estate-price-prediction")

# 3. Load dataset
print("Loading housing dataset...")

df = pd.read_csv(DATA_PATH)

target = "median_house_value_100k"
X = df.drop(columns=[target])
y = df[target]

print(f"Dataset rows: {len(df)}")

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# 5. Build model pipeline
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    (
        "regressor",
        RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    ),
])

# 6. Train model
print("\nTraining house-price prediction model...")
model.fit(X_train, y_train)

# 7. Evaluate model
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\nModel evaluation results")
print("------------------------")
print(f"MAE:  {mae:.4f} (in $100,000 units)")
print(f"RMSE: {rmse:.4f} (in $100,000 units)")
print(f"R2:   {r2:.4f}")

# 8. Log experiment to MLflow
print("\nLogging experiment to MLflow...")

with mlflow.start_run(run_name="random-forest-v1"):
    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 15)
    mlflow.log_param("min_samples_leaf", 2)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("dataset_rows", len(df))

    mlflow.log_metric("mae", float(mae))
    mlflow.log_metric("rmse", float(rmse))
    mlflow.log_metric("r2", float(r2))

    mlflow.sklearn.log_model(
    model,
    name="price_prediction_model",
    serialization_format="pickle",
)

    print("Model metrics logged to MLflow successfully.")

# 9. Save trained model
joblib.dump(model, MODEL_PATH)
print(f"\nModel saved to: {MODEL_PATH}")

# 10. Test a sample prediction
sample = X_test.iloc[[0]]
predicted_value = model.predict(sample)[0]
actual_value = y_test.iloc[0]

print("\nSample prediction")
print("-----------------")
print(f"Predicted median value: ${predicted_value * 100000:,.0f}")
print(f"Actual median value:    ${actual_value * 100000:,.0f}")

print("\nTraining pipeline completed successfully!")
