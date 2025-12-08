# =============================================
# RQ2 MODELING — Predict Volume Using
# Make + Model Year + State
# =============================================

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
df = pd.read_csv("data/cleaned/make_model_year_state_volume.csv")

print("Dataset shape:", df.shape)
print(df.head())


# 3. SELECT FEATURES FOR RQ2
features_rq2 = ["make", "year", "state_clean"]
target = "total_sold_by_make_model_year_state"

rq2_df = df[features_rq2 + [target]].dropna()
print("After cleaning NA:", rq2_df.shape)

X = rq2_df[features_rq2]
y = rq2_df[target]


# 4. PREPROCESSING
categorical_features = ["make", "state_clean"]
numeric_features     = ["year"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)


# 5. BUILD MODELS
dt_model = Pipeline(steps=[
    ("prep", preprocessor),
    ("model", DecisionTreeRegressor(
        max_depth=15,
        random_state=42
    ))
])

rf_model = Pipeline(steps=[
    ("prep", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=150,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ))
])


# 6. TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


# 7. TRAIN MODELS
print("Training Decision Tree...")
dt_model.fit(X_train, y_train)

print("Training Random Forest...")
rf_model.fit(X_train, y_train)

dt_preds = dt_model.predict(X_test)
rf_preds = rf_model.predict(X_test)


# 8. EVALUATION FUNCTION
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


# 9. VISUALIZATION FUNCTION (Volume Specific)
def plot_volume_results(y_test, dt_preds, rf_preds):
    # --- PLOT 1: Scatter Plot (Actual vs Predicted) ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    def draw_scatter(ax, preds, title, color):
        ax.scatter(y_test, preds, alpha=0.4, color=color, label='Predicted')
        min_val = min(y_test.min(), preds.min())
        max_val = max(y_test.max(), preds.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')
        ax.set_xlabel("Actual Volume")
        ax.set_ylabel("Predicted Volume")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    draw_scatter(axes[0], dt_preds, "Decision Tree: Volume Prediction", "orange")
    draw_scatter(axes[1], rf_preds, "Random Forest: Volume Prediction", "purple")
    
    plt.tight_layout()
    plt.show()

    # --- PLOT 2: Line Plot (First 50 Samples) ---
    n_points = 50
    plt.figure(figsize=(14, 5))
    
    y_test_reset = y_test.reset_index(drop=True)[:n_points]
    dt_preds_sub = dt_preds[:n_points]
    rf_preds_sub = rf_preds[:n_points]
    
    plt.plot(y_test_reset, label="Actual Volume", color="black", linewidth=2, linestyle="-")
    plt.plot(dt_preds_sub, label="Decision Tree", color="orange", linestyle="--", alpha=0.7)
    plt.plot(rf_preds_sub, label="Random Forest", color="purple", linestyle="--", alpha=0.7)
    
    plt.title(f"Volume Prediction: Actual vs Predicted (First {n_points} Cases)")
    plt.xlabel("Sample Index")
    plt.ylabel("Volume Sold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# 10. EXECUTE RESULTS
evaluate("Decision Tree (RQ2)", y_test, dt_preds)
evaluate("Random Forest (RQ2)", y_test, rf_preds)

plot_volume_results(y_test, dt_preds, rf_preds)

