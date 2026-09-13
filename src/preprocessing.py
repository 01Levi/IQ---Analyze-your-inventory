from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "sales_lag_1", "sales_lag_3", "sales_lag_7", "sales_lag_14", "sales_lag_30",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_30",
    "rolling_std_7", "rolling_std_30",
    "day_of_week", "day_of_month", "month", "weekend",
    "promotion", "price", "price_change", "trend",
    "product_code",
]

TARGET_COLUMN = "units_sold"

WEEKEND_DAYS = (4, 5)


def clean_sales(sales: pd.DataFrame) -> pd.DataFrame:
    df = sales.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce").fillna(0)
    df["units_sold"] = df["units_sold"].clip(lower=0)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["promotion"] = pd.to_numeric(df["promotion"], errors="coerce").fillna(0).astype(int)

    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)

    full_frames = []
    for pid, g in df.groupby("product_id", sort=False):
        idx = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        g = g.set_index("date").reindex(idx)
        g.index.name = "date"
        g["product_id"] = pid
        g["product_name"] = g["product_name"].ffill().bfill()
        g["category"] = g["category"].ffill().bfill()
        g["units_sold"] = g["units_sold"].fillna(0)
        g["price"] = g["price"].ffill().bfill()
        g["promotion"] = g["promotion"].fillna(0).astype(int)
        full_frames.append(g.reset_index())

    out = pd.concat(full_frames, ignore_index=True)
    return out.sort_values(["product_id", "date"]).reset_index(drop=True)


def add_features(sales: pd.DataFrame) -> pd.DataFrame:
    df = sales.sort_values(["product_id", "date"]).copy()

    g = df.groupby("product_id", sort=False)["units_sold"]

    for lag in (1, 3, 7, 14, 30):
        df[f"sales_lag_{lag}"] = g.shift(lag)

    shifted = g.shift(1)
    df["_shifted"] = shifted
    gs = df.groupby("product_id", sort=False)["_shifted"]

    for window in (7, 14, 30):
        df[f"rolling_mean_{window}"] = gs.transform(
            lambda s, w=window: s.rolling(w, min_periods=2).mean()
        )
    for window in (7, 30):
        df[f"rolling_std_{window}"] = gs.transform(
            lambda s, w=window: s.rolling(w, min_periods=2).std()
        )
    df = df.drop(columns=["_shifted"])

    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["weekend"] = df["day_of_week"].isin(WEEKEND_DAYS).astype(int)

    df["price_change"] = df.groupby("product_id", sort=False)["price"].pct_change().fillna(0)

    df["trend"] = (df["rolling_mean_7"] / df["rolling_mean_30"].replace(0, np.nan))
    df["trend"] = df["trend"].replace([np.inf, -np.inf], np.nan).fillna(1.0)

    codes = {pid: i for i, pid in enumerate(sorted(df["product_id"].unique()))}
    df["product_code"] = df["product_id"].map(codes).astype(int)

    return df.reset_index(drop=True)


def build_training_table(sales: pd.DataFrame) -> pd.DataFrame:
    df = add_features(clean_sales(sales))
    df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    return df


def time_based_split(df: pd.DataFrame, test_days: int = 21) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = df["date"].max() - pd.Timedelta(days=test_days - 1)
    train = df[df["date"] < cutoff].copy()
    test = df[df["date"] >= cutoff].copy()
    return train, test


def make_future_feature_row(history: list[float], future_date: pd.Timestamp,
                            price: float, promotion: int, prev_price: float,
                            product_code: int) -> dict:
    h = np.asarray(history, dtype=float)

    def lag(k: int) -> float:
        return float(h[-k]) if len(h) >= k else float(h.mean())

    def roll_mean(w: int) -> float:
        window = h[-w:] if len(h) >= 2 else h
        return float(np.mean(window)) if len(window) else 0.0

    def roll_std(w: int) -> float:
        window = h[-w:] if len(h) >= 2 else h
        return float(np.std(window, ddof=1)) if len(window) > 1 else 0.0

    mean_7, mean_30 = roll_mean(7), roll_mean(30)
    trend = mean_7 / mean_30 if mean_30 > 0 else 1.0

    return {
        "sales_lag_1": lag(1), "sales_lag_3": lag(3), "sales_lag_7": lag(7),
        "sales_lag_14": lag(14), "sales_lag_30": lag(30),
        "rolling_mean_7": mean_7, "rolling_mean_14": roll_mean(14),
        "rolling_mean_30": mean_30,
        "rolling_std_7": roll_std(7), "rolling_std_30": roll_std(30),
        "day_of_week": int(future_date.dayofweek),
        "day_of_month": int(future_date.day),
        "month": int(future_date.month),
        "weekend": int(future_date.dayofweek in WEEKEND_DAYS),
        "promotion": int(promotion),
        "price": float(price),
        "price_change": float((price - prev_price) / prev_price) if prev_price else 0.0,
        "trend": float(trend),
        "product_code": int(product_code),
    }
