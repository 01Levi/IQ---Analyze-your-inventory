from __future__ import annotations

import numpy as np
import pandas as pd

SERVICE_LEVEL_Z = {
    0.90: 1.28,
    0.95: 1.65,
    0.97: 1.88,
    0.99: 2.33,
}

RISK_ORDER = {"حرج": 0, "مرتفع": 1, "متوسط": 2, "منخفض": 3}

def classify_risk(prob: float) -> str:
    if prob >= 0.70:
        return "حرج"
    if prob >= 0.40:
        return "مرتفع"
    if prob >= 0.15:
        return "متوسط"
    return "منخفض"


def z_for_service_level(service_level: float) -> float:
    key = min(SERVICE_LEVEL_Z, key=lambda k: abs(k - service_level))
    return SERVICE_LEVEL_Z[key]


def monte_carlo_stockout(daily_forecast: np.ndarray, sigma: float,
                         inventory_position: float, n_sims: int = 3000,
                         seed: int = 42) -> tuple[float, float]:
    if len(daily_forecast) == 0:
        return 0.0, float("inf")

    rng = np.random.default_rng(seed)
    sigma = max(float(sigma), 0.1)

    demand = rng.normal(loc=daily_forecast, scale=sigma, size=(n_sims, len(daily_forecast)))
    demand = np.clip(demand, 0, None)
    cumulative = np.cumsum(demand, axis=1)

    breached = cumulative > max(inventory_position, 0)
    stockout_happened = breached.any(axis=1)
    probability = float(stockout_happened.mean())

    if stockout_happened.any():
        first_day = np.argmax(breached[stockout_happened], axis=1) + 1
        days_to_stockout = float(np.mean(first_day))
    else:
        days_to_stockout = float(len(daily_forecast))

    return probability, days_to_stockout


def compute_inventory_plan(products: pd.DataFrame,
                           sales: pd.DataFrame,
                           forecast: pd.DataFrame,
                           residual_std: dict,
                           service_level: float = 0.95,
                           review_period_days: int = 14,
                           overstock_days_threshold: int = 60) -> pd.DataFrame:
    z = z_for_service_level(service_level)
    rows = []

    sales = sales.copy()
    sales["date"] = pd.to_datetime(sales["date"])
    last_date = sales["date"].max()
    recent_cut = last_date - pd.Timedelta(days=29)
    prev_cut = last_date - pd.Timedelta(days=59)

    for _, p in products.iterrows():
        pid = p["product_id"]
        g = sales[sales["product_id"] == pid].sort_values("date")
        fc = forecast[forecast["product_id"] == pid].sort_values("horizon_day")

        lead_time = int(p["lead_time_days"])
        daily_fc = fc["forecast"].to_numpy()

        avg_daily_demand = float(np.mean(daily_fc[:max(lead_time, 7)])) if len(daily_fc) else 0.0
        demand_lead_time = float(np.sum(daily_fc[:lead_time]))
        demand_30d = float(np.sum(daily_fc[:30]))
        demand_7d = float(np.sum(daily_fc[:7]))
        demand_14d = float(np.sum(daily_fc[:14]))

        hist_recent = g[g["date"] >= recent_cut]["units_sold"].to_numpy(dtype=float)
        hist_std = float(np.std(hist_recent, ddof=1)) if len(hist_recent) > 1 else 1.0
        model_std = float(residual_std.get(pid, hist_std))
        demand_std = max(hist_std, model_std, 0.5)

        safety_stock = z * demand_std * np.sqrt(lead_time)
        reorder_point = demand_lead_time + safety_stock
        inventory_position = float(p["on_hand"] + p["on_order"] - p["reserved"])

        days_of_inventory = (inventory_position / avg_daily_demand
                             if avg_daily_demand > 0 else 999.0)

        horizon = max(lead_time, 7)
        prob, sim_days = monte_carlo_stockout(
            daily_fc[:horizon], demand_std, inventory_position
        )
        risk = classify_risk(prob)

        cover_days = min(lead_time + review_period_days, len(daily_fc))
        target_level = float(np.sum(daily_fc[:cover_days])) + safety_stock

        needs_reorder = inventory_position <= reorder_point
        raw_qty = max(0.0, target_level - inventory_position) if needs_reorder else 0.0

        moq = int(p["min_order_qty"])
        if raw_qty <= 0:
            recommended_qty = 0
        else:
            recommended_qty = int(np.ceil(raw_qty / moq) * moq) if moq > 1 else int(np.ceil(raw_qty))

        excess_units = max(0.0, inventory_position - (demand_30d + safety_stock))
        is_overstock = (days_of_inventory > overstock_days_threshold) and (excess_units > 0)
        excess_value = excess_units * float(p["unit_cost"])

        sold_7d = float(g[g["date"] >= last_date - pd.Timedelta(days=6)]["units_sold"].sum())
        sold_14d = float(g[g["date"] >= last_date - pd.Timedelta(days=13)]["units_sold"].sum())
        sold_30d = float(hist_recent.sum())

        recent_avg = float(hist_recent.mean()) if len(hist_recent) else 0.0
        prev_window = g[(g["date"] >= prev_cut) & (g["date"] < recent_cut)]["units_sold"]
        sold_prev_30d = float(prev_window.sum())
        prev_avg = float(prev_window.mean()) if len(prev_window) else 0.0
        demand_change_pct = (100.0 * (recent_avg - prev_avg) / prev_avg) if prev_avg > 0 else 0.0

        shortfall_units = max(0.0, demand_lead_time - inventory_position)
        lost_revenue = shortfall_units * float(p["sale_price"])

        if risk in ("حرج", "مرتفع") or (days_of_inventory < lead_time):
            status = "نفاد وشيك"
        elif needs_reorder:
            status = "يحتاج طلب قريباً"
        elif is_overstock:
            status = "مخزون راكد"
        else:
            status = "آمن"

        rows.append({
            "product_id": pid,
            "product_name": p["product_name"],
            "category": p["category"],
            "supplier": p["supplier"],
            "on_hand": int(p["on_hand"]),
            "on_order": int(p["on_order"]),
            "reserved": int(p["reserved"]),
            "inventory_position": inventory_position,
            "lead_time_days": lead_time,
            "min_order_qty": moq,
            "unit_cost": float(p["unit_cost"]),
            "sale_price": float(p["sale_price"]),
            "avg_daily_demand": avg_daily_demand,
            "recent_avg_sales": recent_avg,
            "sold_7d": sold_7d,
            "sold_14d": sold_14d,
            "sold_30d": sold_30d,
            "sold_prev_30d": sold_prev_30d,
            "demand_change_pct": demand_change_pct,
            "shortfall_units": shortfall_units,
            "lost_revenue": lost_revenue,
            "demand_std": demand_std,
            "demand_7d": demand_7d,
            "demand_14d": demand_14d,
            "demand_30d": demand_30d,
            "lead_time_demand": demand_lead_time,
            "safety_stock": safety_stock,
            "reorder_point": reorder_point,
            "days_of_inventory": days_of_inventory,
            "days_until_stockout": sim_days,
            "stockout_probability": prob,
            "stockout_risk": risk,
            "needs_reorder": bool(needs_reorder),
            "cover_days": int(cover_days),
            "review_period_days": int(review_period_days),
            "target_level": target_level,
            "recommended_order_qty": recommended_qty,
            "order_value": recommended_qty * float(p["unit_cost"]),
            "is_overstock": bool(is_overstock),
            "excess_units": excess_units,
            "excess_value": excess_value,
            "stock_value": inventory_position * float(p["unit_cost"]),
            "status": status,
        })

    plan = pd.DataFrame(rows)
    plan["risk_order"] = plan["stockout_risk"].map(RISK_ORDER)
    return plan.sort_values(["risk_order", "days_until_stockout"]).reset_index(drop=True)


STATUS_SCORE = {
    "آمن": 100,
    "يحتاج طلب قريباً": 72,
    "مخزون راكد": 55,
    "نفاد وشيك": 18,
}


def inventory_health_score(plan: pd.DataFrame) -> int:
    if plan.empty:
        return 0
    weights = (plan["demand_30d"] * plan["sale_price"]).clip(lower=1)
    scores = plan["status"].map(STATUS_SCORE).fillna(60)
    return int(round(float(np.average(scores, weights=weights))))


def portfolio_summary(plan: pd.DataFrame) -> dict:
    return {
        "health": inventory_health_score(plan),
        "critical": int((plan["status"] == "نفاد وشيك").sum()),
        "reorder": int((plan["status"] == "يحتاج طلب قريباً").sum()),
        "overstock": int((plan["status"] == "مخزون راكد").sum()),
        "healthy": int((plan["status"] == "آمن").sum()),
        "total_products": int(len(plan)),
        "stock_value": float(plan["stock_value"].sum()),
        "excess_value": float(plan["excess_value"].sum()),
        "order_value": float(plan["order_value"].sum()),
        "revenue_at_risk": float(
            (plan.loc[plan["status"] == "نفاد وشيك", "demand_30d"] *
             plan.loc[plan["status"] == "نفاد وشيك", "sale_price"]).sum()
        ),
    }


if __name__ == "__main__":
    from src.data_source import DemoCSVDataSource
    from src.forecasting import forecast_future, train_forecast_model

    src = DemoCSVDataSource()
    sales_df, products_df = src.get_sales(), src.get_products()
    result = train_forecast_model(sales_df)
    fc = forecast_future(sales_df, result, horizon=30)
    plan_df = compute_inventory_plan(products_df, sales_df, fc, result.residual_std)

    print(plan_df[["product_name", "on_hand", "avg_daily_demand", "days_of_inventory",
                   "reorder_point", "stockout_probability", "stockout_risk",
                   "recommended_order_qty", "status"]].round(2).to_string(index=False))
    print("\nملخص:", portfolio_summary(plan_df))
