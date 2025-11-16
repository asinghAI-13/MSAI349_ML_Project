# =============================================
# RQ2 MODELING — Predict Volume Using
# Make + Model Year + State
#
# Can ML learn patterns between which brands
# sold most per volume and model year within each state?
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


# 2. LOAD DATASET
df = pd.read_csv("data/cleaned/make_model_year_state_volume.csv")

print("Dataset shape:", df.shape)
print(df.head())


# 3. SELECT FEATURES FOR RQ2
# Inputs: brand (make), model year, state
# Target: total_sold_by_make_model_year_state
features_rq2 = ["make", "year", "state_clean"]
target = "total_sold_by_make_model_year_state"

rq2_df = df[features_rq2 + [target]].dropna()
print("After cleaning NA:", rq2_df.shape)

X = rq2_df[features_rq2]
y = rq2_df[target]


# 4. PREPROCESSING: CATEGORICAL VS NUMERIC
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
dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

dt_preds = dt_model.predict(X_test)
rf_preds = rf_model.predict(X_test)


# 8. EVALUATION FUNCTION
def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    print(f"\n===== {name} =====")
    print("MAE :", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R²  :", round(r2, 4))


# 9. SHOW RESULTS
evaluate("Decision Tree (RQ2)", y_test, dt_preds)
evaluate("Random Forest (RQ2)", y_test, rf_preds)
