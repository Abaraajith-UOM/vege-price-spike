"""Vegetable Price Spike Early Warning: Streamlit front end for the pickled model (AML Final Assignment).

Run:  streamlit run app.py
Loads model.pkl (a dict holding the full pipeline + metadata). Users enter recent prices and rainfall;
the pipeline derives every model feature itself, and SHAP explains each prediction.
"""
import pickle
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))           # the pickle references prep.py / models.py

st.set_page_config(page_title="Vegetable Price Spike Early Warning", page_icon="🥬", layout="centered")

CSS = """
<style>
.block-container {max-width: 980px; padding-top: 1.2rem; padding-bottom: 3rem;}
header[data-testid="stHeader"] {background: transparent;}
.hero {border-radius: 22px; padding: 34px 38px; color: #fff; margin-bottom: 18px;
  background:
    radial-gradient(circle at 88% 20%, rgba(255,255,255,.16) 0, rgba(255,255,255,0) 38%),
    radial-gradient(circle at 70% 110%, rgba(160,190,235,.28) 0, rgba(255,255,255,0) 45%),
    linear-gradient(120deg, #0f2444 0%, #1d4f85 58%, #3b7bb8 100%);}
.hero h1 {font-size: 2.05rem; margin: 0 0 6px 0; color: #fff; line-height: 1.15; padding: 0;}
.hero p {margin: 4px 0; font-size: 1.04rem; color: #e6eefa;}
.hero small {color: #b9cbe3; font-size: .84rem;}
div[data-testid="stVerticalBlockBorderWrapper"], div[class*="st-key-card"] {background: #fff !important; border-radius: 18px !important;
  border: 1px solid #d9e2ee !important; box-shadow: 0 1px 2px rgba(15,36,68,.06); padding: 6px 10px 10px 10px;}
.card-title {font-weight: 650; font-size: 1.02rem; color: #13294b; margin: 0 0 2px 0;}
.card-sub {font-size: .88rem; color: #5b6f86; margin: 0 0 6px 0;}
.big {font-size: 3.4rem; font-weight: 750; line-height: 1; color: #13294b;}
.big small {font-size: 1rem; font-weight: 500; color: #5b6f86; margin-left: 8px;}
.badge {display: inline-block; padding: 6px 14px; border-radius: 999px; font-weight: 650; font-size: .9rem;}
.badge.alert {background: #fff1d6; color: #9a4b00; border: 1px solid #f3cf8f;}
.badge.ok {background: #e3edfa; color: #1d4f85; border: 1px solid #bcd2ee;}
.chip {display: inline-block; padding: 6px 13px; border-radius: 999px; background: #e8eff8; color: #13294b;
  font-size: .86rem; margin: 6px 8px 0 0;}
.gauge {position: relative; height: 14px; border-radius: 999px; background: #e1e8f1; margin: 20px 0 30px 0;}
.gauge .fill {position: absolute; left: 0; top: 0; bottom: 0; border-radius: 999px; background: linear-gradient(90deg,#7fa8d6,#1d4f85);}
.gauge .mark {position: absolute; top: -5px; width: 2px; height: 24px; background: #64748b;}
.gauge .mark.thr {background: #b45309;}
.gauge .lab {position: absolute; top: 26px; transform: translateX(-50%); font-size: .74rem; color: #5b6f86; white-space: nowrap;}
.gauge .lab.thr {color: #b45309;}
.tile {background: #eef3fa; border-radius: 14px; padding: 14px 16px;}
.tile .v {font-size: 1.7rem; font-weight: 720; color: #13294b; line-height: 1.1;}
.tile .l {font-size: .82rem; color: #5b6f86;}
.note {font-size: .84rem; color: #667a90; margin-top: 14px;}
div[data-baseweb="select"] > div {border-radius: 12px;}
button[kind="primary"], button[kind="primaryFormSubmit"] {border-radius: 12px; font-weight: 600;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def load_bundle():
    with open(ROOT / "model.pkl", "rb") as f:
        return pickle.load(f)


B = load_bundle()
pipe, thr = B["pipeline"], B["threshold"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
LABEL = lambda s: s.replace("_", " ")
FEATURE_NAMES = {"item": "Which vegetable", "market": "Market", "price_type": "Retail or wholesale", "lp": "Price level",
                 "r7": "Change vs last week", "r14": "Change vs 2 weeks ago", "r28": "Change vs 4 weeks ago",
                 "prev7": "Change in the week before", "gap4": "Gap to 4-point average", "vol4": "Recent volatility",
                 "rain_ne_7d_mm": "Rain, Nuwara Eliya", "rain_pol_7d_mm": "Rain, Polonnaruwa", "month": "Month"}

st.markdown(
    f"""<div class="hero"><h1>Vegetable Price Spike Early Warning</h1>
<p>Will a vegetable's price rise <b>{B['spike_threshold']:.0%} or more</b> within the next <b>{B['horizon_days']} days</b> in a Colombo-area market?</p>
<small>Daily price reports of the Central Bank of Sri Lanka, 2023 to 2026, plus local rainfall · Model: {B['model_name']}</small></div>""",
    unsafe_allow_html=True)

tab_pred, tab_perf = st.tabs(["Predict", "Model performance"])

with tab_pred:
    with st.container(border=True, key="card1"):
        st.markdown('<div class="card-title">1 · Choose what to check</div>'
                    '<div class="card-sub">Pick an item, market and price type. Quick picks are the items that spike most often.</div>',
                    unsafe_allow_html=True)
        quick = [i for i in ["Tomato", "Beans", "Brinjal", "Green_Chilli", "Carrot", "Lime"] if i in B["items"]]
        default_item = "Beans" if "Beans" in B["items"] else B["items"][0]
        if hasattr(st, "pills"):
            pick = st.pills("Quick pick", quick, format_func=LABEL, selection_mode="single", default=None, label_visibility="collapsed")
        else:
            pick = None
        c1, c2, c3 = st.columns(3)
        item = c1.selectbox("Item", B["items"], index=B["items"].index(pick or default_item), format_func=LABEL, key=f"item_{pick}")
        market = c2.selectbox("Market", B["markets"], format_func=str.title)
        ptype = c3.selectbox("Price type", B["price_types"], format_func=str.title)
    key = (item, market, ptype)
    if key not in B["latest"]:
        st.warning("The training data has no price series for this item / market / price-type combination. Pick another.")
        st.stop()
    d = B["latest"][key]
    k = "|".join(key)                              # new widget keys per selection -> defaults refresh

    with st.container(border=True, key="card2"):
        st.markdown('<div class="card-title">2 · Recent prices and rainfall</div>'
                    f'<div class="card-sub">Pre-filled with the latest real values in the data (report of {d["date"]}). Edit them to try a scenario.</div>',
                    unsafe_allow_html=True)
        with st.form("inputs", border=False):
            a, b_, c_, e = st.columns(4)
            p0 = a.number_input("Today (Rs)", 1.0, 10000.0, d["price_now"], key=f"p0{k}")
            p7 = b_.number_input("7 days ago", 1.0, 10000.0, d["price_7d_ago"], key=f"p7{k}")
            p14 = c_.number_input("14 days ago", 1.0, 10000.0, d["price_14d_ago"], key=f"p14{k}")
            p28 = e.number_input("28 days ago", 1.0, 10000.0, d["price_28d_ago"], key=f"p28{k}")
            r1, r2, r3 = st.columns(3)
            rne = r1.number_input("Rain, last 7 days, Nuwara Eliya (mm)", 0.0, 1500.0, d["rain_ne_7d_mm"], key=f"rne{k}")
            rpo = r2.number_input("Rain, last 7 days, Polonnaruwa (mm)", 0.0, 1500.0, d["rain_pol_7d_mm"], key=f"rpo{k}")
            mdef = pd.Timestamp(d["date"]).month - 1
            month = r3.selectbox("Month", range(1, 13), index=mdef, format_func=lambda m: MONTHS[m - 1], key=f"m{k}")
            go = st.form_submit_button("Check the price outlook", type="primary")

    if go:
        row = pd.DataFrame([{"item": item, "market": market, "price_type": ptype, "price_now": p0, "price_7d_ago": p7,
                             "price_14d_ago": p14, "price_28d_ago": p28, "rain_ne_7d_mm": rne, "rain_pol_7d_mm": rpo,
                             "month": month}])[B["raw_features"]]
        prob = float(pipe.predict_proba(row)[0, 1])
        usual = B.get("item_spike_rate", {}).get(item)
        alert = prob >= thr
        avg4 = np.mean([p0, p7, p14, p28])

        with st.container(border=True, key="card3"):
            st.markdown('<div class="card-title">3 · Outlook</div>'
                        f'<div class="card-sub">{LABEL(item)} · {market.title()} · {ptype}</div>', unsafe_allow_html=True)
            left, right = st.columns([3, 2])
            left.markdown(f'<div class="big">{prob:.0%}<small>chance of a rise of {B["spike_threshold"]:.0%} or more within {B["horizon_days"]} days</small></div>',
                          unsafe_allow_html=True)
            badge = (f'<span class="badge alert">Alert: above the {thr:.0%} warning line</span>' if alert
                     else f'<span class="badge ok">No alert: below the {thr:.0%} warning line</span>')
            right.markdown(f'<div style="text-align:right;padding-top:8px">{badge}</div>', unsafe_allow_html=True)
            marks = f'<div class="mark thr" style="left:{thr * 100:.1f}%"></div><div class="lab thr" style="left:{thr * 100:.1f}%">warning line {thr:.0%}</div>'
            if usual is not None:
                marks += f'<div class="mark" style="left:{usual * 100:.1f}%"></div><div class="lab" style="left:{usual * 100:.1f}%;top:-22px">usual for {LABEL(item)} {usual:.0%}</div>'
            st.markdown(f'<div class="gauge"><div class="fill" style="width:{min(prob, 1) * 100:.1f}%"></div>{marks}</div>', unsafe_allow_html=True)
            chips = [f"{p0 / p7 - 1:+.1%} vs a week ago", f"{p0 / p28 - 1:+.1%} vs four weeks ago", f"{p0 / avg4 - 1:+.1%} vs its 4-point average"]
            if usual is not None:
                chips.append(f"{prob / max(usual, 1e-6):.1f}x the usual rate for {LABEL(item)}")
            st.markdown("".join(f'<span class="chip">{c}</span>' for c in chips), unsafe_allow_html=True)
            if usual is not None and prob < usual:
                st.markdown('<div class="note" style="margin-bottom:6px">Frequent spikers trigger the alert often. A probability below the usual rate means this week looks calmer than normal for this item, even if the alert is on.</div>',
                            unsafe_allow_html=True)

        with st.container(border=True, key="card4"):
            st.markdown('<div class="card-title">4 · Why this prediction?</div>'
                        '<div class="card-sub">SHAP shows how much each input pushed this result up (towards a spike) or down.</div>',
                        unsafe_allow_html=True)
            try:
                import explain
                agg, _, base = explain.shap_by_feature(pipe, row)
                contrib = agg.iloc[0]
                top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(7)
                df = pd.DataFrame({"Input": [FEATURE_NAMES.get(i, i) for i in top.index], "Push": top.values,
                                   "Direction": np.where(top.values > 0, "Raises spike chance", "Lowers spike chance")})
                chart = (alt.Chart(df).mark_bar(cornerRadiusEnd=4, height=22)
                         .encode(x=alt.X("Push:Q", title="Push on the result (log-odds; right = towards a spike)"),
                                 y=alt.Y("Input:N", sort=list(df["Input"]), title=None, axis=alt.Axis(labelOverlap=False, labelLimit=240)),
                                 color=alt.Color("Direction:N", legend=alt.Legend(orient="bottom", title=None),
                                                 scale=alt.Scale(domain=["Raises spike chance", "Lowers spike chance"], range=["#e08a00", "#94a3b8"])),
                                 tooltip=["Input", alt.Tooltip("Push:Q", format="+.2f"), "Direction"])
                         .properties(height=44 * len(df)))
                st.altair_chart(chart, use_container_width=True)
                st.markdown('<div class="note" style="margin-bottom:6px">The pushes add up (with a base value) to the model output. They describe what the model uses, not what causes prices to move.</div>',
                            unsafe_allow_html=True)
            except Exception as e:                      # keep the app usable if shap is missing
                st.info(f"Explanation unavailable here ({type(e).__name__}). Install shap to see it.")

    st.markdown('<div class="note">Planning estimate from past price reports, not an official forecast or trading advice. '
                'Prices move for reasons the data do not contain, such as fuel, transport and imports.</div>', unsafe_allow_html=True)

with tab_perf:
    rep = B.get("report")
    if rep is not None:
        me = rep.get(B["model_name"], {})
        with st.container(border=True, key="card5"):
            st.markdown('<div class="card-title">How well does it work?</div>'
                        f'<div class="card-sub">{B["model_name"]} on the 2026 test year, which was never used for training or tuning.</div>',
                        unsafe_allow_html=True)
            cols = st.columns(4)
            for col, (lab, k_) in zip(cols, [("ROC-AUC", "AUC"), ("Spikes caught (recall)", "recall"),
                                              ("Alerts that were real (precision)", "precision"), ("F1 score", "F1")]):
                if k_ in me:
                    col.markdown(f'<div class="tile"><div class="v">{me[k_]:.2f}</div><div class="l">{lab}</div></div>', unsafe_allow_html=True)
            st.markdown('<div class="note">About 19% of weeks are spikes. The alert catches most of them but only about a third of alerts are real, so use it to decide where to look first, not as a certainty.</div>',
                        unsafe_allow_html=True)
        with st.container(border=True, key="card6"):
            st.markdown('<div class="card-title">All models compared (test 2026)</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(rep).T.apply(lambda c: c.map(lambda v: "–" if v is None or pd.isna(v) else f"{v:.3f}")))
        imp = B.get("shap_importance")
        if imp is not None:
            with st.container(border=True, key="card7"):
                st.markdown('<div class="card-title">What drives the predictions overall</div>'
                            '<div class="card-sub">Mean absolute SHAP value per input.</div>', unsafe_allow_html=True)
                s = pd.Series(imp).sort_values(ascending=False)
                dfi = pd.DataFrame({"Input": [FEATURE_NAMES.get(i, i) for i in s.index], "Importance": s.values})
                st.altair_chart(alt.Chart(dfi).mark_bar(cornerRadiusEnd=4, color="#2a5a8a", height=18)
                                .encode(x=alt.X("Importance:Q", title=None), y=alt.Y("Input:N", sort=list(dfi["Input"]), title=None, axis=alt.Axis(labelOverlap=False, labelLimit=240)),
                                        tooltip=["Input", alt.Tooltip("Importance:Q", format=".3f")]).properties(height=32 * len(dfi)),
                                use_container_width=True)
    with st.container(border=True, key="card8"):
        st.markdown('<div class="card-title">Known limits</div>', unsafe_allow_html=True)
        st.markdown(
            "- Predicts the **direction and size of a 7-day price move**, not the exact price.\n"
            "- Much of the skill is knowing which vegetables are volatile; within one item the ranking is modest.\n"
            "- Alerts are **not equally good for every market**: Narahenpita (retail only) gets fewer alerts and about half of its spikes are caught, against roughly 85% in Dambulla.\n"
            "- Trained on 2023 to 2026 reports; retrain regularly as new reports arrive.\n"
            "- A teaching project: **not** a trading or policy tool.")
