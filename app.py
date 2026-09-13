from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_source import get_data_source
from src.forecasting import forecast_future, train_forecast_model
from src.inventory import compute_inventory_plan, portfolio_summary
from src.recommendations import (build_recommendation, dwd, explain_product, num,
                                 days as fmt_days, store_recommendation)
from src.seasonality import all_season_plans

APP_NAME = "اي كيو — محلل المخزون الذكي"
STORE_NAME = "أرياف للعطور"

SERVICE_LEVEL = 0.95
REVIEW_PERIOD = 14

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🧴",
    layout="wide",
    initial_sidebar_state="expanded",
)

INK = "#2E1B36"
AMBER = "#C08A2E"
BG = "#F6F4F8"
SURFACE = "#FFFFFF"
CRITICAL = "#B02A37"
WARNING = "#C77700"
OK = "#10775C"
INFO = "#4A5CA8"
MUTED = "#7A7285"

STATUS_COLORS = {
    "نفاد وشيك": CRITICAL,
    "يحتاج طلب قريباً": WARNING,
    "مخزون راكد": INFO,
    "آمن": OK,
}
URGENCY_COLORS = {"critical": CRITICAL, "warning": WARNING, "info": INFO, "ok": OK}
TIMING_COLORS = {
    "اطلب الآن": CRITICAL,
    "متأخر — اطلب اليوم": CRITICAL,
    "قريب": WARNING,
    "لاحقاً": OK,
    "فات الموسم": MUTED,
}

def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"], [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {{
            font-family: 'IBM Plex Sans Arabic', 'Segoe UI', Tahoma, sans-serif;
        }}

        [data-testid="stAppViewContainer"], .stApp {{ direction: rtl; }}
        [data-testid="stSidebar"] {{ direction: rtl; text-align: right; }}
        [data-testid="stSidebarCollapsedControl"] {{ right: 1rem; left: auto; }}
        [data-testid="stMarkdownContainer"], .stMarkdown, p, li, label, h1, h2, h3, h4 {{
            text-align: right;
        }}
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
        [data-testid="stMetricDelta"] {{ direction: rtl; text-align: right; }}

        .stApp {{ background: {BG}; }}
        [data-testid="stSidebar"] > div:first-child {{ background: {INK}; }}
        [data-testid="stSidebar"] * {{ color: #F3EFF6 !important; }}
        [data-testid="stSidebar"] .stButton button {{
            background: {AMBER}; color: {INK} !important; border: none;
            font-weight: 600; width: 100%; border-radius: 10px; padding: .65rem;
            font-size: 1rem;
        }}
        [data-testid="stSidebar"] .stButton button:hover {{ background: #D9A342; }}

        [data-testid="stSidebar"] [data-baseweb="select"] > div {{
            background: #43284E !important; border: 1px solid #6B4C77 !important;
            border-radius: 10px;
        }}
        [data-testid="stSidebar"] [data-baseweb="select"] svg {{ fill: {AMBER} !important; }}
        [data-testid="stSidebar"] [data-baseweb="select"] * {{
            color: #FFFFFF !important; font-weight: 600;
        }}
        [data-baseweb="popover"] [role="option"] {{ color: {INK} !important; }}
        [data-baseweb="popover"] li {{ direction: rtl; text-align: right; }}

        .brand-tag {{
            display: inline-block; background: rgba(192,138,46,.22); color: {AMBER} !important;
            border: 1px solid rgba(192,138,46,.5); border-radius: 999px;
            padding: 2px 12px; font-size: .72rem; font-weight: 600; margin-top: 6px;
        }}

        .calc {{ background: {SURFACE}; border-radius: 14px; padding: 8px 20px 16px 20px;
                 border: 1px solid #E4DEEA; }}
        .calc .step {{ display: flex; gap: 12px; padding: 13px 0;
                       border-bottom: 1px dashed #EAE4EF; }}
        .calc .step:last-child {{ border-bottom: none; }}
        .calc .n {{ flex: 0 0 26px; height: 26px; border-radius: 50%;
                    background: {INK}; color: #fff; font-size: .8rem; font-weight: 700;
                    display: flex; align-items: center; justify-content: center; }}
        .calc .q {{ font-weight: 700; color: {INK}; font-size: .95rem; }}
        .calc .a {{ color: #4A4453; font-size: .92rem; line-height: 1.95; margin-top: 2px; }}
        .calc .eq {{ background: #F6F2F8; border-radius: 8px; padding: 4px 10px;
                     display: inline-block; margin-top: 5px; font-size: .88rem;
                     color: {INK}; }}
        .calc .final {{ background: rgba(192,138,46,.12); border-radius: 10px;
                        padding: 12px 14px; margin-top: 10px; color: {INK};
                        font-weight: 700; font-size: 1rem; }}

        .kpi {{
            background: {SURFACE}; border-radius: 14px; padding: 18px 20px;
            border-top: 4px solid var(--accent, {AMBER});
            box-shadow: 0 1px 3px rgba(46,27,54,.08); height: 100%;
        }}
        .kpi .num {{ font-size: 2.1rem; font-weight: 700; color: {INK}; line-height: 1.1; }}
        .kpi .lbl {{ font-size: .92rem; color: {MUTED}; margin-top: 4px; }}
        .kpi .sub {{ font-size: .8rem; color: {MUTED}; margin-top: 8px; }}

        .hero {{
            background: linear-gradient(120deg, {INK} 0%, #45274F 100%);
            border-radius: 18px; padding: 26px 30px; color: #fff;
        }}
        .hero .score {{ font-size: 3.6rem; font-weight: 700; line-height: 1; }}
        .hero .of {{ font-size: 1.1rem; opacity: .65; }}
        .hero .verdict {{ font-size: 1.05rem; margin-top: 10px; opacity: .92; }}
        .hero .meta {{ font-size: .85rem; opacity: .72; margin-top: 14px; line-height: 1.9; }}

        .rec {{
            background: {SURFACE}; border-radius: 12px; padding: 16px 20px 10px 20px;
            border-right: 5px solid var(--accent, {AMBER}); margin-bottom: 4px;
            box-shadow: 0 1px 2px rgba(46,27,54,.06);
        }}
        .rec .title {{ font-weight: 700; color: {INK}; font-size: 1.02rem; }}
        .rec .action {{ color: var(--accent, {AMBER}); font-weight: 700; margin: 8px 0;
                        font-size: 1.05rem; }}
        .rec .body {{ color: #4A4453; font-size: .93rem; line-height: 2; }}

        .badge {{
            display: inline-block; padding: 3px 12px; border-radius: 999px;
            font-size: .78rem; font-weight: 600; color: #fff;
        }}
        .ai-box {{
            background: {SURFACE}; border-radius: 16px; padding: 22px 26px;
            border: 1px solid #E4DEEA; box-shadow: 0 2px 8px rgba(46,27,54,.07);
        }}
        .ai-box h4 {{ color: {INK}; margin: 0 0 6px 0; }}
        .ai-box .line {{ color: #4A4453; line-height: 2.1; font-size: .95rem;
                         margin-bottom: 6px; }}
        .season {{
            background: {SURFACE}; border-radius: 14px; padding: 18px 22px;
            border: 1px solid #E4DEEA; margin-bottom: 14px;
        }}
        .season .name {{ font-size: 1.15rem; font-weight: 700; color: {INK}; }}
        .season .when {{ color: {MUTED}; font-size: .88rem; margin-top: 4px; }}

        .stTabs [data-baseweb="tab-list"] {{ direction: rtl; gap: 6px; }}
        .stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
        header[data-testid="stHeader"] {{ background: transparent; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_css()

@st.cache_data(show_spinner="جاري تحميل بيانات المتجر…")
def load_store_data(source: str = "demo"):
    ds = get_data_source(source)
    sales = ds.get_sales()
    products = ds.get_products()
    ds.validate(sales, products)
    return sales, products

@st.cache_resource(show_spinner="جاري تدريب نموذج التنبؤ على بيانات المبيعات…")
def run_model(_sales: pd.DataFrame, signature: str, horizon: int = 30):
    result = train_forecast_model(_sales)
    forecast = forecast_future(_sales, result, horizon=horizon)
    return result, forecast

@st.cache_data(show_spinner="جاري حساب خطة المخزون…")
def build_plan(_products, _sales, _forecast, _residual_std,
               review_period: int, signature: str):
    return compute_inventory_plan(
        _products, _sales, _forecast, _residual_std,
        service_level=SERVICE_LEVEL, review_period_days=review_period,
    )

sales_df, products_df = load_store_data("demo")
signature = f"{len(sales_df)}|{sales_df['date'].max():%Y-%m-%d}"
model_result, forecast_df = run_model(sales_df, signature)

with st.sidebar:
    st.markdown(
        f"<h2 style='color:{AMBER};margin-bottom:0;font-size:1.45rem'>🧴 {APP_NAME}</h2>"
        f"<p style='opacity:.8;font-size:.9rem;margin:6px 0 0 0'>{STORE_NAME}</p>"
        f"<span class='brand-tag'>إصدار تجريبي</span>",
        unsafe_allow_html=True,
    )

    st.write("")
    if st.button("🤖 توصية الذكاء الاصطناعي", width="stretch"):
        st.session_state["show_ai"] = True

    st.divider()
    st.markdown("**🔍 اختيار المنتج**")
    product_names = products_df["product_name"].tolist()
    selected_product = st.selectbox("المنتج", product_names, label_visibility="collapsed")

    st.divider()
    st.caption(f"البيانات حتى {sales_df['date'].max():%Y-%m-%d}")

plan_df = build_plan(products_df, sales_df, forecast_df, model_result.residual_std,
                     REVIEW_PERIOD, signature)
summary = portfolio_summary(plan_df)

def kpi_card(number, label, sub, color):
    st.markdown(
        f"<div class='kpi' style='--accent:{color}'>"
        f"<div class='num'>{number}</div>"
        f"<div class='lbl'>{label}</div>"
        f"<div class='sub'>{sub}</div></div>",
        unsafe_allow_html=True,
    )

def badge(text, color):
    return f"<span class='badge' style='background:{color}'>{text}</span>"

def money(v):
    return f"{v:,.0f} ريال"

def calc_steps(row, fut) -> str:
    lead = int(row["lead_time_days"])
    daily = row["avg_daily_demand"]
    moq = int(row["min_order_qty"])
    qty = int(row["recommended_order_qty"])
    cover_days = min(lead + REVIEW_PERIOD, len(fut))
    demand_cover = float(fut.head(cover_days)["forecast"].sum())
    position = row["inventory_position"]
    gap = row["reorder_point"] - position

    def step(n, question, answer, equation=None):
        eq = f"<div class='eq'>{equation}</div>" if equation else ""
        return (f"<div class='step'><div class='n'>{n}</div><div>"
                f"<div class='q'>{question}</div>"
                f"<div class='a'>{answer}{eq}</div></div></div>")

    steps = [
        step(1, "كم تبيع من هذا المنتج يومياً؟",
             f"النموذج قرأ مبيعاتك الأخيرة ويتوقع بيع <b>{num(daily)} وحدة في اليوم</b> "
             f"خلال الفترة القادمة."),
        step(2, "كم ستبيع وأنت تنتظر وصول الشحنة؟",
             f"المورد يحتاج <b>{dwd(lead)}</b> ليوصل البضاعة. نجمع توقع كل يوم منها على حدة "
             f"(أيام نهاية الأسبوع أعلى عادة) فيكون الطلب المتوقع خلال فترة الانتظار:",
             f"مجموع توقع الـ {dwd(lead)} القادمة = <b>{num(row['lead_time_demand'])} وحدة</b>"),
        step(3, "وماذا لو ارتفع الطلب فجأة؟",
             f"مبيعاتك تتذبذب بحدود <b>± {num(row['demand_std'])} وحدة</b> يومياً، فنضيف "
             f"كمية احتياطية تحميك من المفاجآت في <b>95% من الحالات</b>:",
             f"مخزون احتياطي = <b>{num(row['safety_stock'])} وحدة</b>"),
        step(4, "متى بالضبط يجب أن تطلب؟",
             "عندما ينزل مخزونك إلى هذا الرقم — لأنه بالضبط ما يكفيك حتى تصل الشحنة:",
             f"{num(row['lead_time_demand'])} + {num(row['safety_stock'])} = "
             f"<b>{num(row['reorder_point'])} وحدة</b> (نقطة الطلب)"),
        step(5, "وكم مخزونك الآن؟",
             f"نحسب المتوفر في المستودع، زائد ما هو في الطريق إليك، ناقص المحجوز لطلبات لم تُشحن:",
             f"{row['on_hand']:,} متوفر + {row['on_order']:,} قيد الشحن − "
             f"{row['reserved']:,} محجوز = <b>{num(position)} وحدة</b>"),
    ]

    if qty > 0:
        raw = max(0.0, demand_cover + row["safety_stock"] - position)
        rounding = (f" ← تقريباً لأقرب مضاعف للحد الأدنى للمورد ({moq} وحدة) = "
                    f"<b>{num(qty)} وحدة</b>") if qty != round(raw) else ""
        steps.append(step(
            6, "إذاً كم تطلب؟",
            f"مخزونك أقل من نقطة الطلب بـ <b>{num(gap)} وحدة</b>. نطلب كمية تكفيك حتى تصل "
            f"الشحنة ({dwd(lead)}) وتستمر حتى مراجعة الشراء القادمة بعد "
            f"{REVIEW_PERIOD} يوم، مع الاحتياطي:",
            f"{num(demand_cover)} طلب متوقع + {num(row['safety_stock'])} احتياطي − "
            f"{num(position)} مخزونك = {num(raw)} وحدة{rounding}"))
        final = (f"✅ التوصية: اطلب <b>{num(qty)} وحدة</b> بتكلفة تقريبية "
                 f"{money(row['order_value'])}")
    else:
        steps.append(step(
            6, "إذاً هل تطلب الآن؟",
            f"لا. مخزونك يزيد عن نقطة الطلب بـ <b>{num(abs(gap))} وحدة</b>، "
            f"أي أنه يكفيك حتى تصل الشحنة القادمة بأمان."))
        final = "✅ التوصية: لا تحتاج طلباً الآن — راجعه في الدورة القادمة"

    return f"<div class='calc'>{''.join(steps)}<div class='final'>{final}</div></div>"

def recommendation_card(rec, row):
    color = URGENCY_COLORS[rec["urgency"]]
    st.markdown(
        f"<div class='rec' style='--accent:{color}'>"
        f"<div class='title'>{rec['product_name']} &nbsp; {badge(rec['headline'], color)}</div>"
        f"<div class='action'>← {rec['action']}</div>"
        f"<div class='body'>{rec['reason']}<br>{rec['risk_note']}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.write("")

if st.session_state.get("show_ai"):
    advice = store_recommendation(plan_df, summary)
    actions_html = "".join(f"<div class='line'>• {a}</div>" for a in advice["actions"])
    st.markdown(
        f"<div class='ai-box'>"
        f"<h4>🤖 توصية الذكاء الاصطناعي لمتجرك</h4>"
        f"<div class='line'><b>{advice['verdict']}</b> — مؤشر صحة المخزون "
        f"{advice['health']}/100.</div>{actions_html}</div>",
        unsafe_allow_html=True,
    )
    if st.button("إخفاء التوصية"):
        st.session_state["show_ai"] = False
        st.rerun()
    st.write("")

tab_overview, tab_recs, tab_product, tab_season = st.tabs(
    ["نظرة عامة", "التوصيات الذكية", "تفاصيل المنتج", "توصيات المواسم"]
)

with tab_overview:
    left, right = st.columns([1.15, 2])

    with right:
        verdict = store_recommendation(plan_df, summary)["verdict"]
        st.markdown(
            f"<div class='hero'>"
            f"<div style='opacity:.7;font-size:.9rem'>مؤشر صحة المخزون</div>"
            f"<div class='score'>{summary['health']}<span class='of'> / 100</span></div>"
            f"<div class='verdict'>{verdict}</div>"
            f"<div class='meta'>قيمة المخزون الحالي {money(summary['stock_value'])} · "
            f"رأس مال مجمّد في منتجات راكدة {money(summary['excess_value'])} · "
            f"مبيعات معرّضة للخطر خلال 30 يوم {money(summary['revenue_at_risk'])}"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    with left:
        counts = plan_df["status"].value_counts()
        fig = go.Figure(go.Bar(
            x=[counts.get(s, 0) for s in STATUS_COLORS],
            y=list(STATUS_COLORS.keys()),
            orientation="h",
            marker_color=list(STATUS_COLORS.values()),
            text=[counts.get(s, 0) for s in STATUS_COLORS],
            textposition="outside",
        ))
        fig.update_layout(
            height=215, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False), yaxis=dict(side="right", tickfont=dict(size=13)),
            font=dict(family="IBM Plex Sans Arabic", color=INK),
        )
        st.plotly_chart(fig, width="stretch", key="status_bar")

    st.write("")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card(summary["critical"], "نفاد وشيك",
                 f"مبيعات معرّضة للخطر {money(summary['revenue_at_risk'])}", CRITICAL)
    with c2:
        kpi_card(summary["reorder"], "يحتاج طلب قريباً",
                 f"قيمة الطلبات المقترحة {money(summary['order_value'])}", WARNING)
    with c3:
        kpi_card(summary["overstock"], "مخزون راكد",
                 f"رأس مال مجمّد {money(summary['excess_value'])}", INFO)
    with c4:
        kpi_card(summary["healthy"], "منتجات آمنة",
                 f"من إجمالي {summary['total_products']} منتج", OK)

    st.write("")
    st.markdown("#### جدول المخزون الكامل")

    view = pd.DataFrame({
        "المنتج": plan_df["product_name"],
        "الحالة": plan_df["status"],
        "المتوفر": plan_df["on_hand"],
        "مبيعات 30 يوم": plan_df["sold_30d"].round(0).astype(int),
        "الطلب اليومي المتوقع": plan_df["avg_daily_demand"].round(0).astype(int),
        "أيام التغطية": plan_df["days_of_inventory"].clip(upper=999).round(0).astype(int),
        "نقطة إعادة الطلب": plan_df["reorder_point"].round(0).astype(int),
        "احتمال النفاد %": (plan_df["stockout_probability"] * 100).round(0),
        "الكمية الموصى بها": plan_df["recommended_order_qty"],
        "تغيّر الطلب %": plan_df["demand_change_pct"].round(0),
    })
    st.dataframe(
        view, width="stretch", hide_index=True, height=430,
        column_config={
            "احتمال النفاد %": st.column_config.ProgressColumn(
                "احتمال النفاد %", min_value=0, max_value=100, format="%d%%"),
            "الكمية الموصى بها": st.column_config.NumberColumn(format="%d"),
        },
    )

    with st.expander("عرض بيانات المبيعات الخام"):
        st.dataframe(sales_df.tail(300), width="stretch", hide_index=True)
        st.download_button("تحميل بيانات المبيعات CSV",
                           sales_df.to_csv(index=False).encode("utf-8-sig"),
                           "sales.csv", "text/csv")

with tab_recs:
    st.markdown("#### التوصيات مرتّبة حسب الأولوية")
    st.caption("كل الأرقام محسوبة من نموذج التنبؤ ومعادلات المخزون — النص شرح مبسّط لها.")

    only_action = st.toggle("إظهار المنتجات التي تحتاج إجراء فقط", value=True)
    recs_source = plan_df[plan_df["status"] != "آمن"] if only_action else plan_df

    if recs_source.empty:
        st.success("لا توجد منتجات تحتاج إجراء الآن.")
    else:
        for _, row in recs_source.iterrows():
            recommendation_card(build_recommendation(row), row)

    orders = plan_df[plan_df["recommended_order_qty"] > 0]
    if len(orders):
        st.markdown("#### أمر الشراء المقترح")
        po = pd.DataFrame({
            "المورد": orders["supplier"],
            "المنتج": orders["product_name"],
            "الكمية": orders["recommended_order_qty"],
            "تكلفة الوحدة": orders["unit_cost"],
            "الإجمالي": orders["order_value"].round(0),
            "مدة التوريد (يوم)": orders["lead_time_days"],
        }).sort_values(["المورد", "الإجمالي"], ascending=[True, False])
        st.dataframe(po, width="stretch", hide_index=True)
        st.markdown(f"**إجمالي أمر الشراء: {money(po['الإجمالي'].sum())}**")
        st.download_button("تحميل أمر الشراء CSV",
                           po.to_csv(index=False).encode("utf-8-sig"),
                           "purchase_order.csv", "text/csv")

with tab_product:
    row = plan_df[plan_df["product_name"] == selected_product].iloc[0]
    pid = row["product_id"]
    hist = sales_df[sales_df["product_id"] == pid].sort_values("date")
    fut = forecast_df[forecast_df["product_id"] == pid].sort_values("horizon_day")

    st.markdown(
        f"### {selected_product} &nbsp; {badge(row['status'], STATUS_COLORS[row['status']])}",
        unsafe_allow_html=True,
    )
    st.caption(f"{row['category']} · المورد: {row['supplier']} · "
               f"مدة التوريد {row['lead_time_days']} يوم · "
               f"الحد الأدنى للطلب {row['min_order_qty']} وحدة · "
               f"سعر البيع {row['sale_price']:,.0f} ريال")

    hz1, hz2 = st.columns([1.1, 3])
    with hz1:
        horizon_choice = st.radio(
            "مدى التنبؤ المعروض", [7, 14, 30], index=2, horizontal=True,
            format_func=lambda d: f"{d} يوم", key="horizon_radio",
        )
    fut_h = fut[fut["horizon_day"] <= horizon_choice]

    st.markdown("##### كم بعت من هذا المنتج؟")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("آخر 7 أيام", f"{num(row['sold_7d'])} وحدة")
    s2.metric("آخر 14 يوم", f"{num(row['sold_14d'])} وحدة")
    s3.metric("آخر 30 يوم", f"{num(row['sold_30d'])} وحدة",
              f"{row['demand_change_pct']:+.0f}% عن الـ 30 يوم السابقة")
    s4.metric("إيراد آخر 30 يوم", money(row["sold_30d"] * row["sale_price"]))

    st.markdown("##### وضع المخزون")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("المخزون المتوفر", f"{row['on_hand']:,} وحدة")
    m2.metric("الطلب اليومي المتوقع", f"{num(row['avg_daily_demand'])} وحدة")
    m3.metric("يكفي لمدة", f"{fmt_days(row['days_of_inventory'])} يوم")
    m4.metric("الكمية الموصى بها", f"{int(row['recommended_order_qty']):,} وحدة")

    show_days = 90
    h = hist.tail(show_days)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=h["date"], y=h["units_sold"], name="المبيعات الفعلية",
        mode="lines", line=dict(color=INK, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=list(fut_h["date"]) + list(fut_h["date"][::-1]),
        y=list(fut_h["forecast_high"]) + list(fut_h["forecast_low"][::-1]),
        fill="toself", fillcolor="rgba(192,138,46,.18)",
        line=dict(color="rgba(0,0,0,0)"), name="النطاق المتوقع", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=fut_h["date"], y=fut_h["forecast"], name="الطلب المتوقع",
        mode="lines", line=dict(color=AMBER, width=3, dash="dot"),
    ))
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=30, b=10),
        title=f"المبيعات آخر {show_days} يوم + التنبؤ لـ {horizon_choice} يوم قادمة",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans Arabic", color=INK),
        legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
        yaxis=dict(title="عدد الوحدات", gridcolor="#E9E4EE", side="right"),
        xaxis=dict(gridcolor="#F1EDF4"),
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch", key="forecast_chart")

    col_a, col_b = st.columns([1.3, 1])

    with col_a:
        st.markdown("##### كيف وصلنا لهذه التوصية؟")
        st.markdown(calc_steps(row, fut), unsafe_allow_html=True)

    with col_b:
        st.markdown("##### مبيعات آخر 8 أسابيع")
        weekly = (hist.set_index("date")["units_sold"]
                  .resample("W").sum().tail(8).reset_index())
        weekly["label"] = weekly["date"].dt.strftime("%m-%d")
        figw = go.Figure(go.Bar(
            x=weekly["label"], y=weekly["units_sold"], marker_color=AMBER,
            text=weekly["units_sold"].astype(int), textposition="outside",
        ))
        figw.update_layout(
            height=250, margin=dict(l=0, r=10, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="IBM Plex Sans Arabic", color=INK),
            yaxis=dict(visible=False), xaxis=dict(tickfont=dict(size=11)),
        )
        st.plotly_chart(figw, width="stretch", key="weekly_chart")

    st.markdown("##### 🤖 شرح الذكاء الاصطناعي")
    st.info(explain_product(row))

with tab_season:
    st.markdown("#### استعد للمواسم قبل أن تفوتك")
    st.caption(
        "الطلب في المواسم يقفز أضعافاً، والمورد يحتاج وقتاً للتوريد. "
        "هنا نحسب لك متى آخر موعد للطلب حتى تصلك البضاعة قبل بداية الموسم."
    )

    season_plans = all_season_plans(plan_df)
    season_tabs = st.tabs([p["name"] for p in season_plans])

    for tab, sp in zip(season_tabs, season_plans):
        with tab:
            when = (f"بعد {sp['days_to_occasion']} يوم"
                    if sp["days_to_occasion"] >= 0 else "مضى هذا العام")
            st.markdown(
                f"<div class='season'><div class='name'>{sp['name']} — {when}</div>"
                f"<div class='when'>تاريخ المناسبة {sp['date']} · "
                f"موسم البيع من {sp['season_start']} إلى {sp['season_end']} "
                f"({sp['season_days']} يوم)</div></div>",
                unsafe_allow_html=True,
            )

            q1, q2, q3 = st.columns(3)
            q1.metric("إيراد إضافي متوقع من الموسم", money(sp["total_extra_revenue"]))
            q2.metric("تكلفة الطلب المقترح", money(sp["total_order_cost"]))
            q3.metric("منتجات يجب طلبها الآن", sp["urgent_count"])

            st.info(f"**{sp['name']}:** {sp['note']}")

            items = sp["items"]
            table = pd.DataFrame({
                "المنتج": items["product_name"],
                "الفئة": items["category"],
                "زيادة الطلب المتوقعة": items["uplift"].map(lambda u: f"×{u:.2f}"),
                "الطلب المعتاد": items["normal_demand"].round(0).astype(int),
                "الطلب في الموسم": items["expected_demand"].round(0).astype(int),
                "وحدات إضافية": items["extra_units"].round(0).astype(int),
                "إيراد إضافي": items["extra_revenue"].round(0).astype(int),
                "الكمية الموصى بها": items["recommended_qty"],
                "آخر موعد للطلب": items["order_deadline"].astype(str),
                "الحالة": items["timing"],
            })
            st.dataframe(table, width="stretch", hide_index=True)

            urgent = items[items["timing"].isin(["اطلب الآن", "متأخر — اطلب اليوم"])]
            if len(urgent):
                top = urgent.iloc[0]
                color = TIMING_COLORS[top["timing"]]
                st.markdown(
                    f"<div class='rec' style='--accent:{color}'>"
                    f"<div class='title'>الأولوية الآن: {top['product_name']} "
                    f"{badge(top['timing'], color)}</div>"
                    f"<div class='action'>← اطلب {num(top['recommended_qty'])} وحدة</div>"
                    f"<div class='body'>الطلب على فئة <b>{top['category']}</b> يرتفع "
                    f"<b>×{top['uplift']:.2f}</b> في {sp['name']}، أي حوالي "
                    f"<b>{num(top['expected_demand'])}</b> وحدة خلال الموسم بدل "
                    f"<b>{num(top['normal_demand'])}</b> وحدة في الأيام العادية. "
                    f"مدة التوريد <b>{int(top['lead_time_days'])} يوم</b>، فإذا طلبت اليوم "
                    f"تصلك الشحنة يوم <b>{top['arrival_date']}</b>.<br>"
                    f"الفرصة الإضافية من هذا المنتج وحده تقارب "
                    f"<b>{money(top['extra_revenue'])}</b>.</div></div>",
                    unsafe_allow_html=True,
                )

            st.caption(
                "معاملات الزيادة الموسمية تقديرات لسوق العطور السعودي وقابلة للتعديل من "
                "ملف `src/seasonality.py`. عند توفر بيانات سنة كاملة سيستخرجها النظام من "
                "مبيعاتك الفعلية. التواريخ الهجرية تقريبية وتخضع لرؤية الهلال."
            )

st.write("")
st.caption(f"{APP_NAME} · إصدار تجريبي · {STORE_NAME} · البيانات مولّدة محلياً وجاهزة لاستبدالها ببيانات متجر سلة لاحقاً.")
