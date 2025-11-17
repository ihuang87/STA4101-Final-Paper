import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneshotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# import data
movies = pd.read_table("/Users/bellehuang/Downloads/STA4101/final_project/movies.txt")

##  Fitting Models 

# LR (predicting tickets.sold)
# Target
movies["Tickets.Sold"] = pd.to_numeric(movies["Tickets.Sold"], errors="coerce")

# Splitting the date up
movies["Release.Date"] = pd.to_datetime(movies["Release.Date"], errors="coerce")
movies["ReleaseYear"]  = movies["Release.Date"].dt.year
movies["ReleaseMonth"] = movies["Release.Date"].dt.month
movies["ReleaseDay"]   = movies["Release.Date"].dt.day

# Predictors
predictors = ["ReleaseYear", "ReleaseMonth", "Distributor", "Genre", "MPAA"]
predictors = [c for c in predictors if c in movies.columns]

X = movies[predictors].copy()
y = movies["Tickets.Sold"]

# Column types
cat_cols = [c for c in ["Distributor", "Genre", "MPAA"] if c in X.columns]
num_cols = [c for c in ["ReleaseYear", "ReleaseMonth"] if c in X.columns]

# Preprocess & model
preprocess = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", OneShotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

pipe = Pipeline(steps=[("prep", preprocess), ("model", LinearRegression())])

# Split the data into train and test, the fit the model on train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipe.fit(X_train, y_train)

y_pred_train = pipe.predict(X_train)
y_pred_test  = pipe.predict(X_test)

metrics = pd.DataFrame({
    "Metric": ["R^2 (train)", "R^2 (test)", "RMSE (train)", "RMSE (test)"],
    "Value": [
        r2_score(y_train, y_pred_train),
        r2_score(y_test, y_pred_test),
        mean_squared_error(y_train, y_pred_train),
        mean_squared_error(y_test, y_pred_test)
    ]
})

# CV R^2
cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="r2")
cv_df = pd.DataFrame({"Fold": range(1, 6), "R^2": cv_scores, "Mean": [cv_scores.mean()]*5, "Std": [cv_scores.std()]*5})

# Coefficients
def get_feature_names(ct, num_cols, cat_cols):
    names = []
    names.extend(num_cols)
    ohe = ct.named_transformers_["cat"]
    if hasattr(ohe, "get_feature_names_out"):
        names.extend(ohe.get_feature_names_out(cat_cols).tolist())
    else:
        for i, base in enumerate(cat_cols):
            names.extend([f"{base}_{c}" for c in ohe.categories_[i]])
    return names

feature_names = get_feature_names(pipe.named_steps["prep"], num_cols, cat_cols)
coefs = pipe.named_steps["model"].coef_
coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs}).sort_values("coef", key=lambda s: s.abs(), ascending=False)

print(cv_df)

cv_rmse = cross_val_score(
    pipe, X, y,
    cv=5,
    scoring="neg_root_mean_squared_error"
)

print("CV RMSE (per fold):", -cv_rmse)
print("CV RMSE mean:", -cv_rmse.mean())

## NN (predicting gross)
# Parse numerics
for c in ["Gross", "Year", "Rank"]:
    if c in movies.columns:
        movies[c] = pd.to_numeric(movies[c], errors="coerce")


# --- Target: log1p(Gross) ---
y_log = np.log1p(movies["Gross"])

# --- Features: EXCLUDE Tickets.Sold ---
feature_cols = [c for c in [
    "Year", "Rank",
    "ReleaseYear", "ReleaseMonth", "ReleaseDay",
    "Distributor", "Genre", "MPAA"
] if c in movies.columns]

X = movies[feature_cols].copy()

# Identify numeric vs categorical
cat_cols = [c for c in ["Distributor", "Genre", "MPAA"] if c in X.columns]
num_cols = [c for c in X.columns if c not in cat_cols]


numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ]
)

# Neural network
mlp = MLPRegressor(
    hidden_layer_sizes=(32,),   # <- single hidden layer, 32 units
    activation="relu",
    solver="adam",
    alpha=1e-4,
    max_iter=200,
    early_stopping=True,
    n_iter_no_change=10,
    random_state=42,
)

pipe = Pipeline(steps=[
    ("prep", preprocess),
    ("model", mlp),
])

# splitting into train and test
X_train, X_test, y_train_log, y_test_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

# Fit NN on log(Gross)
pipe.fit(X_train, y_train_log)

# Predict
y_pred_train_log = pipe.predict(X_train)
y_pred_test_log  = pipe.predict(X_test)

# Back-transform to original Gross scale
y_train = np.expm1(y_train_log)
y_test  = np.expm1(y_test_log)
y_pred_train = np.expm1(y_pred_train_log)
y_pred_test  = np.expm1(y_pred_test_log)

# Metrics on original Gross scale 
r2_train  = r2_score(y_train, y_pred_train)
r2_test   = r2_score(y_test,  y_pred_test)
rmse_train = mean_squared_error(y_train, y_pred_train)
rmse_test  = mean_squared_error(y_test,  y_pred_test)

print("\n=== Original Gross scale ===")
print("R^2 train:",  r2_train)
print("R^2 test:",   r2_test)
print("RMSE train:", rmse_train)
print("RMSE test:",  rmse_test)


# NN (predicting gross with tickets.sold)

# --- Target: log1p(Gross) ---
y_log = np.log1p(movies["Gross"])

# --- Features: NOW INCLUDING Tickets.Sold ---
feature_cols = [c for c in [
    "Tickets.Sold",          # <- added back in
    "Year", "Rank",
    "ReleaseYear", "ReleaseMonth", "ReleaseDay",
    "Distributor", "Genre", "MPAA"
] if c in movies.columns]

X = movies[feature_cols].copy()

# Identify numeric vs categorical
cat_cols = [c for c in ["Distributor", "Genre", "MPAA"] if c in X.columns]
num_cols = [c for c in X.columns if c not in cat_cols]

# --- Preprocessing ---
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ]
)

# --- Neural net (same size: 1 hidden layer, 32 units) ---
mlp = MLPRegressor(
    hidden_layer_sizes=(32,),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    max_iter=200,
    early_stopping=True,
    n_iter_no_change=10,
    random_state=42,
)

pipe = Pipeline(steps=[
    ("prep", preprocess),
    ("model", mlp),
])

# spliting data
X_train, X_test, y_train_log, y_test_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

# Fit on log(Gross)
pipe.fit(X_train, y_train_log)

# Predict
y_pred_train_log = pipe.predict(X_train)
y_pred_test_log  = pipe.predict(X_test)

# Back-transform
y_train = np.expm1(y_train_log)
y_test  = np.expm1(y_test_log)
y_pred_train = np.expm1(y_pred_train_log)
y_pred_test  = np.expm1(y_pred_test_log)

# --- Metrics on original scale ---
r2_train  = r2_score(y_train, y_pred_train)
r2_test   = r2_score(y_test,  y_pred_test)
rmse_train = mean_squared_error(y_train, y_pred_train)
rmse_test  = mean_squared_error(y_test,  y_pred_test)

print("\n=== Original Gross scale ===")
print("R^2 train:",  r2_train)
print("R^2 test:",   r2_test)
print("RMSE train:", rmse_train)
print("RMSE test:",  rmse_test)

#LR (predicting gross with tickets.sold)
# Target
y = movies["Gross"]

# Use ONLY numeric predictors (including Tickets.Sold) for speed/stability
feature_cols = [c for c in [
    "Tickets.Sold",
    "Year", "Rank",
    "ReleaseYear", "ReleaseMonth", "ReleaseDay"
] if c in movies.columns]

X = movies[feature_cols].copy()

# Impute numeric missing values
imp = SimpleImputer(strategy="median")
X_imp = imp.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_imp, y, test_size=0.2, random_state=42
)

# Fit linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict
y_pred_train = lr.predict(X_train)
y_pred_test  = lr.predict(X_test)

# Metrics
r2_train  = r2_score(y_train, y_pred_train)
r2_test   = r2_score(y_test,  y_pred_test)
rmse_train = mean_squared_error(y_train, y_pred_train)
rmse_test  = mean_squared_error(y_test,  y_pred_test)

(r2_train, r2_test, rmse_train, rmse_test)
