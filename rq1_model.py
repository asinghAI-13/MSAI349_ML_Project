# =============================================
# RQ1 MODELING — Predict Selling Price Using
# Model Year + Brand + Condition + Odometer + Other Features
# =============================================

# 1. IMPORTS
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 2. LOAD DATASET (VS Code path)
df = pd.read_csv("data/cleaned/vehicle_model_core.csv")

print("Dataset shape:", df.shape)
print(df.head())


# 3. SELECT FEATURES FOR RQ1
# Model year is the main categorical feature, but we include 
# additional predictors for realistic performance.
features_rq1 = [
    "year",             # main feature for RQ1
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
if len(rq1_df) > 100000:
    rq1_df = rq1_df.sample(100000, random_state=42)
print("Using sample size:", rq1_df.shape)


# 4. SPLIT FEATURES & TARGET
X = rq1_df[features_rq1]
y = rq1_df[target]


# 5. SEPARATE CATEGORICAL VS NUMERIC FEATURES
categorical_features = ["make", "body_group", "color_group", "state_clean"]
numeric_features     = ["year", "condition_std", "odometer_clean", "mmr_clean"]

# OneHot encode categoricals, keep numerics as-is
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
        max_depth=12,
        random_state=42
    ))
])

rf_model = Pipeline(steps=[
    ("prep", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=100,     # DEFAULT RF SIZE (FAST + ACCURATE)
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
dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Predictions
dt_preds = dt_model.predict(X_test)
rf_preds = rf_model.predict(X_test)


# 9. EVALUATION FUNCTION (FIXED)
def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    print(f"\n===== {name} =====")
    print("MAE :", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R²  :", round(r2, 4))


# 10. SHOW RESULTS
evaluate("Decision Tree (RQ1)", y_test, dt_preds)
evaluate("Random Forest (RQ1)", y_test, rf_preds)

