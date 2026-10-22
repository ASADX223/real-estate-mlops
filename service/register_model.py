
from pathlib import Path

import bentoml
import bentoml.sklearn
import joblib

# Locate project and trained model
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "price_model.pkl"

# Load the existing trained model
print("Loading trained Random Forest model...")
model = joblib.load(MODEL_PATH)

# Register model in BentoML model store
print("Registering model with BentoML...")

model_info = bentoml.sklearn.save_model(
    "real_estate_price_model",
    model,
)

print("\nModel registered successfully!")
print(f"Model tag: {model_info.tag}")
