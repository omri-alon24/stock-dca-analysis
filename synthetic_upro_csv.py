import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# -----------------------------
# 1. Download data
# -----------------------------
spy = yf.download("SPY", start="1993-01-01", auto_adjust=True)["Close"]
upro = yf.download("UPRO", start="2009-01-01", auto_adjust=True)["Close"]

# Daily percent returns
spy_ret = spy.pct_change().dropna()
upro_ret = upro.pct_change().dropna()

# Align
returns = pd.concat([spy_ret, upro_ret], axis=1)
returns.columns = ["spy_ret", "upro_ret"]
returns = returns.dropna()

# -----------------------------
# 2. Build lagged features
# -----------------------------
X = returns[["spy_ret"]]
y = returns["upro_ret"]

# Add lag features for RF/MLP
for lag in [1, 2, 3]:
    X[f"spy_ret_lag{lag}"] = returns["spy_ret"].shift(lag)

X = X.dropna()
y = y.loc[X.index]

# -----------------------------
# 3. Train multiple models
# -----------------------------
ts_split = TimeSeriesSplit(n_splits=5)
models = {
    "naive_3x": None,
    "linear": LinearRegression(),
    "poly2": (PolynomialFeatures(degree=2), LinearRegression()),
    "rf": RandomForestRegressor(n_estimators=200, random_state=42),
    "mlp": MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=500, random_state=42)
}

results = {}

for name, model in models.items():
    mse_scores = []
    for train_idx, test_idx in ts_split.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if name == "naive_3x":
            y_pred = 3 * X_test["spy_ret"]
        elif name == "poly2":
            poly, reg = model
            X_train_poly = poly.fit_transform(X_train[["spy_ret"]])
            X_test_poly = poly.transform(X_test[["spy_ret"]])
            reg.fit(X_train_poly, y_train)
            y_pred = reg.predict(X_test_poly)
        else:
            model.fit(X_train.fillna(0), y_train)
            y_pred = model.predict(X_test.fillna(0))

        mse_scores.append(mean_squared_error(y_test, y_pred))

    results[name] = np.mean(mse_scores)

best_model_name = min(results, key=results.get)
print("Best model:", best_model_name)

# -----------------------------
# 4. Refit best model on full overlap period
# -----------------------------
if best_model_name == "naive_3x":
    def predict_fn(df):
        return 3 * df["spy_ret"]
elif best_model_name == "poly2":
    poly, reg = models["poly2"]
    X_poly = poly.fit_transform(X[["spy_ret"]])
    reg.fit(X_poly, y)
    def predict_fn(df):
        return reg.predict(poly.transform(df[["spy_ret"]]))
else:
    model = models[best_model_name]
    model.fit(X.fillna(0), y)
    def predict_fn(df):
        return model.predict(df.fillna(0))

# -----------------------------
# 5. Backcast synthetic UPRO
# -----------------------------
full_spy_ret = spy.pct_change().dropna()
X_full = pd.DataFrame({"spy_ret": full_spy_ret})
for lag in [1, 2, 3]:
    X_full[f"spy_ret_lag{lag}"] = full_spy_ret.shift(lag)

synthetic_upro_ret = pd.Series(predict_fn(X_full.fillna(0)), index=full_spy_ret.index)

# -----------------------------
# 6. Convert to synthetic price series
# -----------------------------
synthetic_price = (1 + synthetic_upro_ret).cumprod() * 100

# Hybrid splice: use real UPRO after 2009
synthetic_price.loc[upro.index] = (1 + upro.pct_change().dropna()).cumprod() * synthetic_price.loc[upro.index[0]] / upro.iloc[0]

# -----------------------------
# 7. Export CSV only
# -----------------------------
export = pd.DataFrame({
    "SPY_Close": spy,
    "SPY_Return": full_spy_ret,
    "Synthetic_UPRO_Return": synthetic_upro_ret,
    "Synthetic_UPRO_Price": synthetic_price
})
export.to_csv("synthetic_upro_daily.csv")
print("CSV saved: synthetic_upro_daily.csv")
