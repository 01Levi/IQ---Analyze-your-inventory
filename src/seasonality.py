from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

BUFFER_DAYS = 3


OCCASIONS = [
    {
        "key": "national_day",
        "name": "اليوم الوطني السعودي",
        "date": date(2026, 9, 23),
        "days_before": 12,
        "days_after": 2,
        "note": "موسم الهدايا والمعطرات، والعروض بالعلم الأخضر تحقق مبيعات عالية.",
        "uplift": {
            "هدايا": 1.60, "معطرات منزل": 1.45, "عطور رجالية": 1.35,
            "عطور نسائية": 1.30, "معطرات جسم": 1.25, "عطور فاخرة": 1.25,
            "عود ودهن": 1.20, "بخور": 1.20, "مسك": 1.15,
        },
    },
    {
        "key": "ramadan",
        "name": "رمضان 1448",
        "date": date(2027, 2, 8),
        "days_before": 14,
        "days_after": 25,
        "note": "ذروة البخور والعود والمسك طوال الشهر، خصوصاً قبل المغرب وفي الليالي.",
        "uplift": {
            "بخور": 2.20, "عود ودهن": 1.90, "مسك": 1.65, "معطرات منزل": 1.55,
            "عطور فاخرة": 1.30, "هدايا": 1.25, "عطور رجالية": 1.20,
            "عطور نسائية": 1.20, "معطرات جسم": 1.15,
        },
    },
    {
        "key": "eid_fitr",
        "name": "عيد الفطر 1448",
        "date": date(2027, 3, 9),
        "days_before": 20,
        "days_after": 3,
        "note": "أقوى موسم للعطور في السنة — الشراء يبدأ من العشر الأواخر.",
        "uplift": {
            "هدايا": 2.60, "عطور فاخرة": 2.40, "عطور نسائية": 2.10,
            "عطور رجالية": 2.00, "عود ودهن": 1.85, "بخور": 1.55,
            "مسك": 1.45, "معطرات جسم": 1.35, "معطرات منزل": 1.30,
        },
    },
    {
        "key": "eid_adha",
        "name": "عيد الأضحى 1448",
        "date": date(2027, 5, 16),
        "days_before": 14,
        "days_after": 3,
        "note": "موسم العود والبخور والهدايا، ويتزامن مع موسم الحج وزيادة الزوار.",
        "uplift": {
            "عود ودهن": 1.95, "هدايا": 1.80, "بخور": 1.75, "عطور فاخرة": 1.65,
            "عطور رجالية": 1.45, "عطور نسائية": 1.45, "مسك": 1.35,
            "معطرات منزل": 1.25, "معطرات جسم": 1.20,
        },
    },
]


def _round_to_moq(qty: float, moq: int) -> int:
    if qty <= 0:
        return 0
    return int(np.ceil(qty / moq) * moq) if moq > 1 else int(np.ceil(qty))


def build_season_plan(plan: pd.DataFrame, occasion: dict,
                      today: date | None = None, top_n: int = 8) -> dict:
    today = today or date.today()

    season_start = occasion["date"] - timedelta(days=occasion["days_before"])
    season_end = occasion["date"] + timedelta(days=occasion["days_after"])
    season_days = occasion["days_before"] + occasion["days_after"] + 1
    days_to_season = (season_start - today).days
    days_to_occasion = (occasion["date"] - today).days

    rows = []
    for _, r in plan.iterrows():
        uplift = occasion["uplift"].get(r["category"], 1.15)
        normal = float(r["avg_daily_demand"]) * season_days
        expected = normal * uplift
        extra = expected - normal

        lead = int(r["lead_time_days"])
        order_deadline = season_start - timedelta(days=lead + BUFFER_DAYS)
        days_to_deadline = (order_deadline - today).days

        consumed_before = float(r["avg_daily_demand"]) * max(days_to_season, 0)
        available_at_season = max(0.0, float(r["inventory_position"]) - consumed_before)
        gap = expected + float(r["safety_stock"]) - available_at_season
        qty = _round_to_moq(gap, int(r["min_order_qty"]))

        arrival = today + timedelta(days=lead)

        if days_to_deadline < 0:
            timing = "متأخر — اطلب اليوم" if arrival <= season_end else "فات الموسم"
        elif days_to_deadline <= 7:
            timing = "اطلب الآن"
        elif days_to_deadline <= 30:
            timing = "قريب"
        else:
            timing = "لاحقاً"

        rows.append({
            "product_id": r["product_id"],
            "product_name": r["product_name"],
            "category": r["category"],
            "supplier": r["supplier"],
            "uplift": uplift,
            "normal_demand": normal,
            "expected_demand": expected,
            "extra_units": extra,
            "extra_revenue": extra * float(r["sale_price"]),
            "available_at_season": available_at_season,
            "recommended_qty": qty,
            "order_cost": qty * float(r["unit_cost"]),
            "lead_time_days": lead,
            "order_deadline": order_deadline,
            "days_to_deadline": days_to_deadline,
            "arrival_date": arrival,
            "timing": timing,
        })

    df = pd.DataFrame(rows).sort_values("extra_revenue", ascending=False).reset_index(drop=True)

    return {
        "name": occasion["name"],
        "note": occasion["note"],
        "date": occasion["date"],
        "season_start": season_start,
        "season_end": season_end,
        "season_days": season_days,
        "days_to_occasion": days_to_occasion,
        "days_to_season": days_to_season,
        "items": df.head(top_n),
        "all_items": df,
        "total_extra_revenue": float(df["extra_revenue"].sum()),
        "total_order_cost": float(df.loc[df["recommended_qty"] > 0, "order_cost"].sum()),
        "urgent_count": int(df["timing"].isin(["اطلب الآن", "متأخر — اطلب اليوم"]).sum()),
    }


def all_season_plans(plan: pd.DataFrame, today: date | None = None) -> list[dict]:
    today = today or date.today()
    plans = [build_season_plan(plan, occ, today) for occ in OCCASIONS]
    plans.sort(key=lambda p: (p["days_to_occasion"] < 0, abs(p["days_to_occasion"])))
    return plans


if __name__ == "__main__":
    from src.data_source import DemoCSVDataSource
    from src.forecasting import forecast_future, train_forecast_model
    from src.inventory import compute_inventory_plan

    s = DemoCSVDataSource()
    sales, products = s.get_sales(), s.get_products()
    res = train_forecast_model(sales)
    fc = forecast_future(sales, res, 30)
    plan_df = compute_inventory_plan(products, sales, fc, res.residual_std)

    for p in all_season_plans(plan_df):
        print(f"\n=== {p['name']} — بعد {p['days_to_occasion']} يوم "
              f"({p['season_start']} إلى {p['season_end']}) ===")
        print(f"إيراد إضافي متوقع: {p['total_extra_revenue']:,.0f} ريال | "
              f"تكلفة الطلب: {p['total_order_cost']:,.0f} ريال | "
              f"عاجل: {p['urgent_count']}")
        print(p["items"][["product_name", "uplift", "extra_units",
                          "recommended_qty", "order_deadline", "timing"]]
              .round(1).head(5).to_string(index=False))
