from __future__ import annotations

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd


DAYS = 180
END_DATE = date.today() - timedelta(days=1)
SEED = 7

WEEKEND_DAYS = (4, 5)


PRODUCT_CATALOG = [
    dict(name="دهن عود كمبودي 6 مل", category="عود ودهن", base=7.5, trend=1.95,
         weekend_boost=1.25, season_amp=0.10, season_phase=120, promo_freq=6,
         promo_lift=2.1, dispersion=8, price=890, cost=520, lead_time=14,
         moq=10, stock_profile="critical", supplier="مورد العود الآسيوي"),

    dict(name="عطر العنبر الملكي 100 مل", category="عطور فاخرة", base=12.0, trend=1.70,
         weekend_boost=1.30, season_amp=0.12, season_phase=140, promo_freq=7,
         promo_lift=2.4, dispersion=10, price=420, cost=210, lead_time=10,
         moq=24, stock_profile="critical", supplier="مصنع العنبر - دبي"),

    dict(name="مسك أبيض رول أون 10 مل", category="مسك", base=26.0, trend=1.45,
         weekend_boost=1.15, season_amp=0.06, season_phase=60, promo_freq=9,
         promo_lift=1.9, dispersion=14, price=45, cost=17, lead_time=7,
         moq=100, stock_profile="critical", supplier="مسك الرياض"),

    dict(name="عطر الورد الطائفي 50 مل", category="عطور نسائية", base=15.0, trend=1.25,
         weekend_boost=1.20, season_amp=0.18, season_phase=95, promo_freq=8,
         promo_lift=2.0, dispersion=12, price=310, cost=160, lead_time=9,
         moq=20, stock_profile="reorder", supplier="ورد الطائف"),

    dict(name="بخور معمول فاخر 100 جم", category="بخور", base=18.0, trend=1.10,
         weekend_boost=1.45, season_amp=0.15, season_phase=150, promo_freq=7,
         promo_lift=2.2, dispersion=9, price=185, cost=90, lead_time=12,
         moq=30, stock_profile="reorder", supplier="بيت البخور"),

    dict(name="عطر الجاذبية للرجال 100 مل", category="عطور رجالية", base=20.0, trend=1.18,
         weekend_boost=1.28, season_amp=0.08, season_phase=110, promo_freq=8,
         promo_lift=2.3, dispersion=11, price=260, cost=125, lead_time=8,
         moq=25, stock_profile="reorder", supplier="مصنع الرياض للعطور"),

    dict(name="عطر الصندل الهندي 75 مل", category="عطور فاخرة", base=9.0, trend=1.02,
         weekend_boost=1.10, season_amp=0.05, season_phase=40, promo_freq=5,
         promo_lift=1.7, dispersion=18, price=350, cost=190, lead_time=11,
         moq=15, stock_profile="healthy", supplier="مورد الصندل الهندي"),

    dict(name="عطر الياسمين 50 مل", category="عطور نسائية", base=13.0, trend=1.00,
         weekend_boost=1.12, season_amp=0.07, season_phase=70, promo_freq=6,
         promo_lift=1.8, dispersion=16, price=175, cost=80, lead_time=7,
         moq=30, stock_profile="healthy", supplier="مصنع الرياض للعطور"),

    dict(name="معطر جسم منعش 200 مل", category="معطرات جسم", base=30.0, trend=1.05,
         weekend_boost=1.18, season_amp=0.20, season_phase=30, promo_freq=10,
         promo_lift=1.9, dispersion=15, price=59, cost=22, lead_time=6,
         moq=120, stock_profile="healthy", supplier="مصنع جدة للمعطرات"),

    dict(name="عطر الزعفران 30 مل", category="عطور فاخرة", base=6.0, trend=1.08,
         weekend_boost=1.15, season_amp=0.10, season_phase=130, promo_freq=5,
         promo_lift=2.0, dispersion=14, price=395, cost=215, lead_time=13,
         moq=12, stock_profile="healthy", supplier="مورد العود الآسيوي"),

    dict(name="بخاخ معطر منزل - عود 300 مل", category="معطرات منزل", base=22.0, trend=1.06,
         weekend_boost=1.22, season_amp=0.12, season_phase=100, promo_freq=8,
         promo_lift=1.85, dispersion=13, price=79, cost=31, lead_time=7,
         moq=100, stock_profile="healthy", supplier="مصنع جدة للمعطرات"),

    dict(name="طقم هدايا عطور فاخر", category="هدايا", base=5.0, trend=1.35,
         weekend_boost=1.60, season_amp=0.45, season_phase=165, promo_freq=11,
         promo_lift=3.2, dispersion=3, price=650, cost=340, lead_time=15,
         moq=10, stock_profile="reorder", supplier="بيت الهدايا"),

    dict(name="عطر التوت البري 50 مل", category="عطور نسائية", base=8.0, trend=1.12,
         weekend_boost=1.35, season_amp=0.30, season_phase=45, promo_freq=12,
         promo_lift=2.8, dispersion=4, price=145, cost=62, lead_time=8,
         moq=40, stock_profile="healthy", supplier="مصنع جدة للمعطرات"),

    dict(name="عطر الفانيلا الحلو 50 مل", category="عطور نسائية", base=17.0, trend=0.45,
         weekend_boost=1.08, season_amp=0.06, season_phase=20, promo_freq=9,
         promo_lift=1.7, dispersion=12, price=130, cost=58, lead_time=7,
         moq=50, stock_profile="overstock", supplier="مصنع جدة للمعطرات"),

    dict(name="عطر المسك الأزرق 100 مل", category="عطور رجالية", base=14.0, trend=0.55,
         weekend_boost=1.10, season_amp=0.05, season_phase=25, promo_freq=8,
         promo_lift=1.8, dispersion=11, price=155, cost=70, lead_time=9,
         moq=50, stock_profile="overstock", supplier="مصنع الرياض للعطور"),

    dict(name="معطر ملابس برائحة الليمون", category="معطرات منزل", base=11.0, trend=0.62,
         weekend_boost=1.05, season_amp=0.08, season_phase=15, promo_freq=7,
         promo_lift=1.6, dispersion=13, price=39, cost=14, lead_time=6,
         moq=150, stock_profile="overstock", supplier="مصنع جدة للمعطرات"),

    dict(name="عطر صيفي منعش 100 مل", category="عطور رجالية", base=16.0, trend=0.70,
         weekend_boost=1.14, season_amp=0.35, season_phase=10, promo_freq=8,
         promo_lift=1.9, dispersion=12, price=120, cost=52, lead_time=8,
         moq=60, stock_profile="overstock", supplier="مصنع الرياض للعطور"),

    dict(name="دهن عود هندي VIP 3 مل", category="عود ودهن", base=1.2, trend=1.15,
         weekend_boost=1.30, season_amp=0.12, season_phase=140, promo_freq=3,
         promo_lift=2.5, dispersion=6, price=1450, cost=880, lead_time=18,
         moq=5, stock_profile="healthy", supplier="مورد العود الآسيوي"),

    dict(name="عطر الأميرة الذهبي 75 مل", category="عطور نسائية", base=2.0, trend=1.05,
         weekend_boost=1.20, season_amp=0.10, season_phase=125, promo_freq=4,
         promo_lift=2.2, dispersion=7, price=520, cost=290, lead_time=12,
         moq=10, stock_profile="reorder", supplier="بيت الهدايا"),

    dict(name="مسك الطهارة 30 مل", category="مسك", base=3.5, trend=0.95,
         weekend_boost=1.05, season_amp=0.06, season_phase=80, promo_freq=4,
         promo_lift=1.6, dispersion=9, price=65, cost=26, lead_time=7,
         moq=60, stock_profile="overstock", supplier="مسك الرياض"),
]


def _trend_factor(t: np.ndarray, total_trend: float) -> np.ndarray:
    return 1.0 + (total_trend - 1.0) * (t / (len(t) - 1))


def _weekly_factor(dow: np.ndarray, weekend_boost: float) -> np.ndarray:
    base_weights = np.array([0.95, 0.92, 0.98, 1.05, 1.00, 1.00, 1.10])
    factor = base_weights[dow]
    is_weekend = np.isin(dow, WEEKEND_DAYS)
    return factor * np.where(is_weekend, weekend_boost, 1.0)


def _season_factor(t: np.ndarray, amp: float, phase: float) -> np.ndarray:
    if amp <= 0:
        return np.ones_like(t, dtype=float)
    return 1.0 + amp * np.sin(2 * np.pi * (t - phase) / 180.0)


def _make_promotions(rng: np.random.Generator, n_days: int, freq: int) -> np.ndarray:
    promo = np.zeros(n_days, dtype=int)
    n_promos = max(1, freq)
    starts = rng.choice(np.arange(5, n_days - 5), size=n_promos, replace=False)
    for s in starts:
        length = int(rng.integers(2, 5))
        promo[s:s + length] = 1
    return promo


def generate_store_data(days: int = DAYS, end_date: date = END_DATE,
                        seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    start_date = end_date - timedelta(days=days - 1)
    dates = pd.date_range(start=start_date, periods=days, freq="D")
    t = np.arange(days)
    dow = dates.dayofweek.to_numpy()

    sales_rows = []
    product_rows = []

    for idx, cfg in enumerate(PRODUCT_CATALOG, start=1):
        product_id = f"P{idx:03d}"

        promo = _make_promotions(rng, days, cfg["promo_freq"])

        lam = (
            cfg["base"]
            * _trend_factor(t, cfg["trend"])
            * _weekly_factor(dow, cfg["weekend_boost"])
            * _season_factor(t, cfg["season_amp"], cfg["season_phase"])
            * np.where(promo == 1, cfg["promo_lift"], 1.0)
        )

        r = float(cfg["dispersion"])
        gamma_noise = rng.gamma(shape=r, scale=1.0 / r, size=days)
        units = rng.poisson(np.maximum(lam * gamma_noise, 0.02))

        price = np.full(days, float(cfg["price"]))
        price = price * np.where(promo == 1, rng.uniform(0.80, 0.90), 1.0)
        if rng.random() < 0.4:
            change_day = int(rng.integers(60, 140))
            price[change_day:] *= float(rng.uniform(0.92, 1.10))
        price = np.round(price, 2)

        sales_rows.append(pd.DataFrame({
            "date": dates,
            "product_id": product_id,
            "product_name": cfg["name"],
            "category": cfg["category"],
            "units_sold": units,
            "price": price,
            "promotion": promo,
        }))

        recent_avg = float(np.mean(units[-21:])) or 0.5
        profile = cfg["stock_profile"]
        lead = cfg["lead_time"]

        if profile == "critical":
            days_cover = rng.uniform(1.5, 3.5)
        elif profile == "reorder":
            days_cover = rng.uniform(lead * 0.7, lead * 1.15)
        elif profile == "healthy":
            days_cover = rng.uniform(lead + 18, lead + 40)
        else:
            days_cover = rng.uniform(95, 190)

        on_hand = int(max(0, round(recent_avg * days_cover)))
        on_order = int(rng.choice([0, 0, 0, cfg["moq"]]))
        reserved = int(rng.integers(0, max(2, int(recent_avg * 0.4) + 1)))

        product_rows.append({
            "product_id": product_id,
            "product_name": cfg["name"],
            "category": cfg["category"],
            "supplier": cfg["supplier"],
            "unit_cost": cfg["cost"],
            "sale_price": cfg["price"],
            "on_hand": on_hand,
            "on_order": on_order,
            "reserved": reserved,
            "lead_time_days": lead,
            "min_order_qty": cfg["moq"],
        })

    sales_df = pd.concat(sales_rows, ignore_index=True)
    sales_df = sales_df.sort_values(["product_id", "date"]).reset_index(drop=True)
    products_df = pd.DataFrame(product_rows)

    return sales_df, products_df


def save_store_data(data_dir: str = "data", **kwargs) -> tuple[str, str]:
    os.makedirs(data_dir, exist_ok=True)
    sales_df, products_df = generate_store_data(**kwargs)

    sales_path = os.path.join(data_dir, "sample_sales.csv")
    products_path = os.path.join(data_dir, "products.csv")

    sales_df.to_csv(sales_path, index=False, encoding="utf-8-sig")
    products_df.to_csv(products_path, index=False, encoding="utf-8-sig")
    return sales_path, products_path


if __name__ == "__main__":
    s_path, p_path = save_store_data()
    sales = pd.read_csv(s_path)
    print("تم إنشاء البيانات بنجاح")
    print(f"  ملف المبيعات : {s_path}  ({len(sales):,} صف)")
    print(f"  ملف المنتجات : {p_path}")
    print(f"  عدد المنتجات : {sales['product_id'].nunique()}")
    print(f"  الفترة       : {sales['date'].min()} إلى {sales['date'].max()}")
