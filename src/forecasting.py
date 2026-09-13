from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_training_table,
    make_future_feature_row,
    time_based_split,
)

warnings.filterwarnings("ignore")

TEST_DAYS = 21
MAX_HORIZON = 30


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(100.0 * np.sum(np.abs(y_true - y_pred)) / denom)


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(y_true)
    if denom == 0:
        return float("nan")
    return float(100.0 * np.sum(y_pred - y_true) / denom)


@dataclass
class ForecastResult:
    model: object
    model_name: str
    metrics: dict
    test_predictions: pd.DataFrame
    residual_std: dict = field(default_factory=dict)
    feature_importance: pd.DataFrame | None = None
    train_rows: int = 0
    test_rows: int = 0


def baseline_moving_average(test: pd.DataFrame) -> np.ndarray:
    return test["rolling_mean_7"].to_numpy()


def baseline_ema(df: pd.DataFrame, alpha: float = 0.35) -> pd.Series:
    return (
        df.groupby("product_id", sort=False)[TARGET_COLUMN]
        .transform(lambda s: s.shift(1).ewm(alpha=alpha, adjust=False).mean())
    )


def _build_model():
    try:
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(
            objective="poisson",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=40,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
        )
        return model, "LightGBM (Poisson)"
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(
            loss="poisson", max_iter=300, learning_rate=0.06,
            max_depth=5, min_samples_leaf=40, random_state=42
        )
        return model, "HistGradientBoosting (scikit-learn)"


def train_forecast_model(sales: pd.DataFrame, test_days: int = TEST_DAYS) -> ForecastResult:
    table = build_training_table(sales)
    train, test = time_based_split(table, test_days=test_days)

    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN]
    X_test, y_test = test[FEATURE_COLUMNS], test[TARGET_COLUMN]

    model, model_name = _build_model()
    model.fit(X_train, y_train)

    pred_ml = np.clip(model.predict(X_test), 0, None)
    pred_ma = np.clip(baseline_moving_average(test), 0, None)

    ema_all = baseline_ema(table)
    pred_ema = np.clip(ema_all.loc[test.index].to_numpy(), 0, None)

    metrics = {
        "ml": {"name": model_name,
               "mae": mae(y_test, pred_ml),
               "wape": wape(y_test, pred_ml),
               "bias": bias(y_test, pred_ml)},
        "moving_average": {"name": "متوسط متحرك 7 أيام",
                           "mae": mae(y_test, pred_ma),
                           "wape": wape(y_test, pred_ma),
                           "bias": bias(y_test, pred_ma)},
        "ema": {"name": "متوسط أسّي EMA",
                "mae": mae(y_test, pred_ema),
                "wape": wape(y_test, pred_ema),
                "bias": bias(y_test, pred_ema)},
    }
    best_baseline = min(metrics["moving_average"]["wape"], metrics["ema"]["wape"])
    metrics["improvement_pct"] = (
        100.0 * (best_baseline - metrics["ml"]["wape"]) / best_baseline
        if best_baseline else 0.0
    )

    test_predictions = test[["date", "product_id", "product_name", TARGET_COLUMN]].copy()
    test_predictions["pred_ml"] = pred_ml
    test_predictions["pred_ma"] = pred_ma
    test_predictions["pred_ema"] = pred_ema
    test_predictions["error"] = test_predictions[TARGET_COLUMN] - test_predictions["pred_ml"]

    residual_std = (
        test_predictions.groupby("product_id")["error"].std().fillna(0).to_dict()
    )
    fallback = table.groupby("product_id")[TARGET_COLUMN].std().to_dict()
    for pid, val in residual_std.items():
        if not np.isfinite(val) or val <= 0:
            residual_std[pid] = float(max(fallback.get(pid, 1.0), 0.5))

    importance = None
    if hasattr(model, "feature_importances_"):
        importance = (
            pd.DataFrame({"feature": FEATURE_COLUMNS,
                          "importance": model.feature_importances_})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    return ForecastResult(
        model=model,
        model_name=model_name,
        metrics=metrics,
        test_predictions=test_predictions,
        residual_std={k: float(v) for k, v in residual_std.items()},
        feature_importance=importance,
        train_rows=len(train),
        test_rows=len(test),
    )


def forecast_future(sales: pd.DataFrame, result: ForecastResult,
                    horizon: int = MAX_HORIZON, z: float = 1.28) -> pd.DataFrame:
    from src.preprocessing import add_features, clean_sales

    clean = clean_sales(sales)
    featured = add_features(clean)
    codes = featured[["product_id", "product_code"]].drop_duplicates().set_index("product_id")

    rows = []
    for pid, g in clean.groupby("product_id", sort=False):
        g = g.sort_values("date")
        history = g[TARGET_COLUMN].astype(float).tolist()
        last_date = g["date"].max()
        last_price = float(g["price"].iloc[-1])
        prev_price = float(g["price"].iloc[-2]) if len(g) > 1 else last_price
        product_name = g["product_name"].iloc[-1]
        code = int(codes.loc[pid, "product_code"])
        sigma = float(result.residual_std.get(pid, 1.0))

        for step in range(1, horizon + 1):
            future_date = last_date + pd.Timedelta(days=step)
            row = make_future_feature_row(
                history=history, future_date=future_date, price=last_price,
                promotion=0, prev_price=prev_price, product_code=code,
            )
            X = pd.DataFrame([row])[FEATURE_COLUMNS]
            yhat = float(np.clip(result.model.predict(X)[0], 0, None))

            spread = z * sigma * np.sqrt(step)
            rows.append({
                "product_id": pid,
                "product_name": product_name,
                "date": future_date,
                "horizon_day": step,
                "forecast": yhat,
                "forecast_low": max(0.0, yhat - spread),
                "forecast_high": yhat + spread,
            })
            history.append(yhat)
            prev_price = last_price

    return pd.DataFrame(rows)


def summarize_forecast(forecast: pd.DataFrame,
                       horizons: tuple[int, ...] = (7, 14, 30)) -> pd.DataFrame:
    out = []
    for pid, g in forecast.groupby("product_id", sort=False):
        g = g.sort_values("horizon_day")
        row = {"product_id": pid, "product_name": g["product_name"].iloc[0]}
        for h in horizons:
            sub = g[g["horizon_day"] <= h]
            row[f"forecast_{h}d"] = float(sub["forecast"].sum())
            row[f"avg_daily_{h}d"] = float(sub["forecast"].mean())
        row["avg_daily_demand"] = row.get("avg_daily_7d", 0.0)
        out.append(row)
    return pd.DataFrame(out)


def save_model(result: ForecastResult, models_dir: str = "models") -> str:
    import joblib
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, "demand_model.joblib")
    joblib.dump({"model": result.model, "metrics": result.metrics,
                 "residual_std": result.residual_std}, path)
    return path


if __name__ == "__main__":
    from src.data_source import DemoCSVDataSource

    src = DemoCSVDataSource()
    sales_df = src.get_sales()

    res = train_forecast_model(sales_df)
    print(f"النموذج: {res.model_name}")
    print(f"صفوف التدريب: {res.train_rows:,} | صفوف الاختبار: {res.test_rows:,}")
    for key in ("ml", "moving_average", "ema"):
        m = res.metrics[key]
        print(f"  {m['name']:<28} MAE={m['mae']:.2f}  WAPE={m['wape']:.1f}%")
    print(f"  تحسّن النموذج عن أفضل Baseline: {res.metrics['improvement_pct']:.1f}%")

    fc = forecast_future(sales_df, res, horizon=30)
    print(f"\nصفوف التنبؤ المستقبلي: {len(fc):,}")
    print(summarize_forecast(fc).head(5).to_string(index=False))
