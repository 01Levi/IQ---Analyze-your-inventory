from __future__ import annotations

import pandas as pd


def num(value: float) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{int(round(float(value))):,}"


def days(value: float) -> str:
    if value is None or pd.isna(value):
        return "—"
    value = float(value)
    if value >= 10:
        return f"{int(round(value)):,}"
    return f"{value:.1f}".rstrip("0").rstrip(".")


def day_word(value: float) -> str:
    if value is None or pd.isna(value):
        return "يوم"
    v = float(value)
    whole = abs(v - round(v)) < 0.05
    if whole and round(v) == 2:
        return "يومين"
    if whole and 3 <= round(v) <= 10:
        return "أيام"
    return "يوم"


def dwd(value: float) -> str:
    word = day_word(value)
    if word == "يومين":
        return "يومين"
    return f"{days(value)} {word}"


def _money(value: float) -> str:
    return f"{value:,.0f} ريال"


def demand_direction(change_pct: float) -> str:
    if change_pct >= 20:
        return f"ارتفع الطلب بنسبة {change_pct:.0f}%"
    if change_pct >= 7:
        return f"الطلب في ارتفاع خفيف ({change_pct:.0f}%)"
    if change_pct <= -20:
        return f"انخفض الطلب بنسبة {abs(change_pct):.0f}%"
    if change_pct <= -7:
        return f"الطلب في تراجع خفيف ({abs(change_pct):.0f}%)"
    return "الطلب مستقر تقريباً"


def sales_evidence(row: pd.Series) -> str:
    return (
        f"بعت منه <b>{num(row['sold_30d'])}</b> وحدة خلال آخر 30 يوم، "
        f"منها <b>{num(row['sold_14d'])}</b> في آخر أسبوعين "
        f"و<b>{num(row['sold_7d'])}</b> في آخر 7 أيام"
    )


def build_recommendation(row: pd.Series) -> dict:
    name = row["product_name"]
    status = row["status"]
    daily = row["avg_daily_demand"]
    lead = int(row["lead_time_days"])
    prob = row["stockout_probability"]
    qty = int(row["recommended_order_qty"])
    change = row["demand_change_pct"]
    evidence = sales_evidence(row)

    if status == "نفاد وشيك":
        if change >= 20:
            headline = f"طلب مرتفع ومتصاعد +{change:.0f}%"
        elif change <= -10:
            headline = "المخزون لا يكفي رغم تراجع الطلب"
        else:
            headline = "المخزون لا يكفي حتى تصل الشحنة"

        action = f"اطلب {num(qty)} وحدة اليوم" if qty > 0 else "راجع المخزون فوراً"
        reason = (
            f"{evidence} — {demand_direction(change)} مقارنة بالشهر الذي قبله. "
            f"المتوقع أن تبيع <b>{num(daily)}</b> وحدة يومياً، ومخزونك "
            f"<b>{num(row['inventory_position'])}</b> وحدة يكفي "
            f"<b>{dwd(row['days_until_stockout'])} فقط</b>، "
            f"بينما شحنتك تحتاج <b>{dwd(lead)}</b> لتصل."
        )
        if row["lost_revenue"] > 0:
            risk_note = (
                f"إن لم تطلب الآن، ستنقصك <b>{num(row['shortfall_units'])}</b> وحدة قبل "
                f"وصول الشحنة، أي مبيعات ضائعة تقارب <b>{_money(row['lost_revenue'])}</b> — "
                f"واحتمال النفاد <b>{prob*100:.0f}%</b>."
            )
        else:
            risk_note = f"احتمال النفاد قبل وصول الشحنة <b>{prob*100:.0f}%</b>."
        urgency = "critical"

    elif status == "يحتاج طلب قريباً":
        headline = (f"وصل نقطة إعادة الطلب والطلب صاعد +{change:.0f}%"
                    if change >= 15 else "وصل إلى نقطة إعادة الطلب")
        action = f"جهّز أمر شراء بـ {num(qty)} وحدة" if qty > 0 else "راقب المنتج"
        reason = (
            f"{evidence} — {demand_direction(change)}. "
            f"مخزونك <b>{num(row['inventory_position'])}</b> وحدة وهو عند نقطة إعادة الطلب "
            f"(<b>{num(row['reorder_point'])}</b> وحدة) المحسوبة على أساس "
            f"<b>{num(daily)}</b> وحدة يومياً ومدة توريد <b>{dwd(lead)}</b>."
        )
        risk_note = (
            f"الطلب الآن يعني وصول البضاعة قبل النفاد. تأجيله أسبوعاً يرفع احتمال "
            f"النفاد من <b>{prob*100:.0f}%</b> ويضطرك لشحن مستعجل أغلى."
        )
        urgency = "warning"

    elif status == "مخزون راكد":
        headline = "رأس مالك متجمّد في هذا المنتج"
        action = "أوقف الشراء وفعّل عرضاً تسويقياً"
        drop = f"{demand_direction(change)}، و" if change <= -7 else "لكن "
        reason = (
            f"{evidence} — {drop}لديك <b>{num(row['inventory_position'])}</b> وحدة "
            f"والمتوقع أن تبيع <b>{num(daily)}</b> وحدة يومياً فقط، أي أن الكمية "
            f"تكفي <b>{dwd(row['days_of_inventory'])}</b>."
        )
        risk_note = (
            f"الكمية الزائدة عن حاجتك <b>{num(row['excess_units'])}</b> وحدة بقيمة "
            f"<b>{_money(row['excess_value'])}</b> — كان يمكن أن تشتري بها منتجات سريعة الحركة."
        )
        urgency = "info"

    else:
        headline = "الوضع آمن — لا تشترِ الآن"
        action = "لا يحتاج إجراء"
        reason = (
            f"{evidence} — {demand_direction(change)}. "
            f"مخزونك يغطي <b>{dwd(row['days_of_inventory'])}</b> مقابل مدة توريد "
            f"<b>{dwd(lead)}</b>، والمتوقع <b>{num(daily)}</b> وحدة يومياً."
        )
        risk_note = f"احتمال النفاد خلال مدة التوريد منخفض (<b>{prob*100:.0f}%</b>)."
        urgency = "ok"

    return {
        "product_id": row["product_id"],
        "product_name": name,
        "headline": headline,
        "action": action,
        "reason": reason,
        "risk_note": risk_note,
        "urgency": urgency,
        "qty": qty,
        "order_value": row["order_value"],
    }


def explain_product(row: pd.Series) -> str:
    rec = build_recommendation(row)

    def clean(text: str) -> str:
        return text.replace("<b>", "**").replace("</b>", "**")

    parts = [f"**{rec['headline']}.** {clean(rec['reason'])} {clean(rec['risk_note'])}"]

    if rec["qty"] > 0:
        parts.append(
            f"**القرار: {rec['action']}.** الكمية محسوبة لتغطية مدة التوريد "
            f"({dwd(int(row['lead_time_days']))}) وفترة المراجعة بعدها، مع مخزون أمان "
            f"{num(row['safety_stock'])} وحدة. تكلفة الطلب تقريباً "
            f"{_money(row['order_value'])} مقابل مبيعات متوقعة "
            f"{_money(row['demand_30d'] * row['sale_price'])} خلال 30 يوم."
        )
    elif row["status"] == "مخزون راكد":
        discount = 15 if row["days_of_inventory"] < 120 else 25
        parts.append(
            f"**القرار: {rec['action']}.** خصم مقترح {discount}% أو ضمّه في باقة هدايا. "
            f"تحريك نصف الكمية الزائدة يعيد لك حوالي "
            f"{_money(row['excess_units'] * 0.5 * row['sale_price'])} سيولة."
        )
    else:
        parts.append(f"**القرار: {rec['action']}.** تابعه في المراجعة القادمة.")

    return "\n\n".join(parts)


def store_recommendation(plan: pd.DataFrame, summary: dict) -> dict:
    health = summary["health"]

    if health >= 85:
        verdict = "مخزون متجرك في وضع جيد جداً"
    elif health >= 70:
        verdict = "الوضع مقبول لكن هناك منتجات تحتاج انتباهك"
    elif health >= 50:
        verdict = "متجرك يحتاج تدخلاً هذا الأسبوع"
    else:
        verdict = "الوضع حرج: منتجات على وشك النفاد وأخرى تجمّد رأس مالك"

    critical = plan[plan["status"] == "نفاد وشيك"].head(3)
    overstock = plan[plan["status"] == "مخزون راكد"].sort_values(
        "excess_value", ascending=False).head(3)

    actions = []

    if len(critical):
        names = "، ".join(critical["product_name"].tolist())
        total_qty = int(critical["recommended_order_qty"].sum())
        total_val = float(critical["order_value"].sum())
        lost = float(critical["lost_revenue"].sum())
        actions.append(
            f"<b>اطلب اليوم:</b> {names} — إجمالي <b>{num(total_qty)}</b> وحدة بتكلفة "
            f"<b>{_money(total_val)}</b>. هذه المنتجات ستنفد قبل وصول شحنتها، "
            f"والمبيعات الضائعة خلال فترة الانتظار وحدها تقارب <b>{_money(lost)}</b>."
        )

    reorder = plan[plan["status"] == "يحتاج طلب قريباً"]
    if len(reorder):
        top = reorder.iloc[0]
        actions.append(
            f"<b>جهّز خلال الأسبوع:</b> {len(reorder)} منتج وصل نقطة إعادة الطلب، "
            f"أبرزها {top['product_name']} (بعت منه <b>{num(top['sold_30d'])}</b> وحدة "
            f"في آخر 30 يوم). الطلب المبكر يجنّبك تكلفة الشحن المستعجل."
        )

    if len(overstock):
        names = "، ".join(overstock["product_name"].tolist())
        actions.append(
            f"<b>حرّك المخزون الراكد:</b> {names} — رأس مال مجمّد يقارب "
            f"<b>{_money(summary['excess_value'])}</b>. عرض ترويجي أو باقة هدايا "
            f"يعيد جزءاً منه سيولة تشتري بها منتجات سريعة الحركة."
        )

    growing = plan[plan["demand_change_pct"] >= 20].sort_values(
        "demand_change_pct", ascending=False).head(2)
    if len(growing):
        items = "، ".join(
            f"{r['product_name']} (+{r['demand_change_pct']:.0f}% — "
            f"{num(r['sold_30d'])} وحدة في 30 يوم)"
            for _, r in growing.iterrows()
        )
        actions.append(
            f"<b>فرصة نمو:</b> ارتفع الطلب على {items}. ارفع كمية الطلب القادم لها "
            f"قبل أن تتحول إلى نفاد."
        )

    return {"verdict": verdict, "health": health, "actions": actions}
