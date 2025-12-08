# RQ1 MODELING — Predict Selling Price Using
# Model Year + Brand + Condition + Odometer + Other Features


# 1. IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 2. LOAD DATASET
# Ensure this path matches your local folder structure
df = pd.read_csv("data/cleaned/vehicle_model_core.csv")

print("Dataset shape:", df.shape)
print(df.head())


# 3. SELECT FEATURES FOR RQ1
features_rq1 = [
    "year",             # main feature
    "make",
    "body_group",
    "color_group",
    "state_clean",
    "condition_std",
    "odometer_clean",
    "mmr_clean"
]

target = "sellingprice_clean"

# Filter rows with complete data
rq1_df = df[features_rq1 + [target]].dropna()
print("After cleaning NA:", rq1_df.shape)

# SAMPLE FOR SPEED (100,000 ROWS)
# If you want full accuracy for the final paper, comment this block out.
if len(rq1_df) > 100000:
    rq1_df = rq1_df.sample(100000, random_state=42)
print("Using sample size:", rq1_df.shape)


# 4. SPLIT FEATURES & TARGET
X = rq1_df[features_rq1]
y = rq1_df[target]


# 5. PREPROCESSING
categorical_features = ["make", "body_group", "color_group", "state_clean"]
numeric_features     = ["year", "condition_std", "odometer_clean", "mmr_clean"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)


# 6. BUILD MODELS
dt_model = Pipeline(steps=[
    ("prep", preprocessor),
    ("model", DecisionTreeRegressor(
        max_depth=10,
        random_state=42
    ))
])

rf_model = Pipeline(steps=[
    ("prep", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ))
])


# 7. TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


# 8. TRAIN MODELS
print("Training Decision Tree...")
dt_model.fit(X_train, y_train)

print("Training Random Forest...")
rf_model.fit(X_train, y_train)

# Predictions
dt_preds = dt_model.predict(X_test)
rf_preds = rf_model.predict(X_test)


# 9. EVALUATION FUNCTION
def evaluate(name, y_true, y_pred):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    print(f"\n===== {name} =====")
    print("MSE :", round(mse, 2))
    print("RMSE:", round(rmse, 2))
    print("MAE :", round(mae, 2))
    print("R²  :", round(r2, 4))


# 10. VISUALIZATION FUNCTION (Side-by-Side)
def plot_results(y_test, dt_preds, rf_preds):
    # --- PLOT 1: Scatter Plot (Actual vs Predicted) ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    def draw_scatter(ax, preds, title, color):
        ax.scatter(y_test, preds, alpha=0.3, color=color, label='Predicted')
        min_val = min(y_test.min(), preds.min())
        max_val = max(y_test.max(), preds.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')
        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Predicted Price")
        ax.set_title(title)
        ax.legend()

    draw_scatter(axes[0], dt_preds, "Decision Tree: Actual vs Predicted", "blue")
    draw_scatter(axes[1], rf_preds, "Random Forest: Actual vs Predicted", "green")
    
    plt.tight_layout()
    plt.show()

    # --- PLOT 2: Line Plot (First 50 Test Samples) ---
    n_points = 50
    plt.figure(figsize=(14, 5))
    
    y_test_reset = y_test.reset_index(drop=True)[:n_points]
    dt_preds_sub = dt_preds[:n_points]
    rf_preds_sub = rf_preds[:n_points]
    
    plt.plot(y_test_reset, label="Actual", color="black", linewidth=2, linestyle="-")
    plt.plot(dt_preds_sub, label="Decision Tree", color="blue", linestyle="--", alpha=0.7)
    plt.plot(rf_preds_sub, label="Random Forest", color="green", linestyle="--", alpha=0.7)
    
    plt.title(f"Model Comparison: First {n_points} Test Cases")
    plt.xlabel("Sample Index")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# 11. EXECUTE RESULTS
evaluate("Decision Tree (RQ1)", y_test, dt_preds)
evaluate("Random Forest (RQ1)", y_test, rf_preds)

plot_results(y_test, dt_preds, rf_preds)

