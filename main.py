# streamlit_app_fixed_v1.3.py — Polished, more robust Streamlit Loan EMI Planner (v1.3)
# Save and run: `streamlit run streamlit_app_fixed_v1.3.py`

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import math
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List, Dict

# -------------------------
# App config
# -------------------------
st.set_page_config(
    page_title="Loan EMI Planner — Rates & Prepayment Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# Themes (light & dark) — simplified and robust CSS
# -------------------------
_LIGHT_THEME = """
:root {
  --bg-primary: #f7f9fc;
  --bg-secondary: #ffffff;
  --bg-card: #ffffff;
  --text-color: #111827;
  --text-muted: #4b5563; /* slightly darker muted text for better contrast */
  --accent: #2563eb;
  --positive: #16a34a;
  --negative: #dc2626;
}
[data-testid="stAppViewContainer"] { background: var(--bg-primary); color: var(--text-color); }
[data-testid="stSidebar"] { background: var(--bg-secondary); }
.ui-card { background: var(--bg-card); border-radius: 12px; padding: 14px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 12px; }
.ui-muted { color: var(--text-muted) !important; }
.ui-metric-label { font-weight: 600; color: var(--text-muted); font-size: 0.95rem; }
.ui-metric-value { font-weight: 700; font-size: 1.4rem; color: var(--accent); }
.ui-badge { padding:4px 10px; border-radius:999px; font-weight:600; font-size:0.85rem; }
.ui-pos { background: rgba(34,197,94,0.1); color: var(--positive); }
.ui-neg { background: rgba(239,68,68,0.1); color: var(--negative); }

section.main > div.block-container { padding-top: 8px; padding-left: 18px; padding-right: 18px; }
.stDataFrame table, .stTable { background: var(--bg-secondary); color: var(--text-color); border-radius: 8px; }

/* enforce theme text colors for all common elements */
[data-testid="stAppViewContainer"] * { color: var(--text-color) !important; }
[data-testid="stSidebar"] * { color: var(--text-color) !important; }

/* links should use accent */
a, a:visited, a:active, .stMarkdown a { color: var(--accent) !important; }

/* muted captions should use text-muted */
.stCaption, .css-1lsmgbg, .stText, .stMarkdown { color: var(--text-muted) !important; }

/* table / dataframe text */
.stDataFrame table td, .stTable td, .stDataFrame table th, .stTable th { color: var(--text-color) !important; }

/* buttons/inputs labels fallback */
button, input, label, .stNumberInput, .stSelectbox { color: var(--text-color) !important; }

/* --- Widget / input specific fixes --- */
/* Number & text inputs (Streamlit widgets) */
[data-testid="stAppViewContainer"] input[type="number"],
[data-testid="stAppViewContainer"] input[type="text"],
[data-testid="stSidebar"] input[type="number"],
[data-testid="stSidebar"] input[type="text"],
.stNumberInput input,
.stTextInput input,
.stTextArea textarea {
  background: var(--bg-secondary) !important;    /* match theme background */
  color: var(--text-color) !important;           /* readable text color per theme */
  caret-color: var(--accent) !important;
  border-color: rgba(0,0,0,0.08) !important;
}

/* Placeholder / faint captions inside inputs */
[data-testid="stAppViewContainer"] input::placeholder,
[data-testid="stSidebar"] input::placeholder,
.stNumberInput input::placeholder,
.stTextInput input::placeholder {
  color: var(--text-muted) !important;
  opacity: 1 !important;
}

/* Select boxes and dropdown value text */
.stSelectbox > div, .stSelectbox button, .streamlit-expanderHeader {
  color: var(--text-color) !important;
}

/* Ensure table cells also have readable text & backgrounds */
.stDataFrame table td, .stTable td, .stDataFrame table th, .stTable th {
  background: transparent !important;
  color: var(--text-color) !important;
}

/* small: numeric spinner buttons keep native appearance but readable */
input[type="number"]::-webkit-inner-spin-button,
input[type="number"]::-webkit-outer-spin-button {
  -webkit-appearance: inner-spin-button;
}

/* --- Fix icons, SVGs and widget labels (plus/minus, dropdown arrows, radio/select/checkbox labels) --- */
/* Force SVG/icon fills and colors to follow theme text color */
[data-testid="stAppViewContainer"] svg,
[data-testid="stSidebar"] svg,
[data-testid="stAppViewContainer"] button svg,
[data-testid="stSidebar"] button svg {
  fill: var(--text-color) !important;
  color: var(--text-color) !important;
}

/* Make the label/text inside common widgets readable */
[data-testid="stAppViewContainer"] .stSelectbox *,
[data-testid="stAppViewContainer"] .stRadio *,
[data-testid="stAppViewContainer"] .stCheckbox *,
[data-testid="stAppViewContainer"] .stNumberInput *,
[data-testid="stAppViewContainer"] .stSlider *,
[data-testid="stSidebar"] .stSelectbox *,
[data-testid="stSidebar"] .stRadio *,
[data-testid="stSidebar"] .stCheckbox *,
[data-testid="stSidebar"] .stNumberInput * {
  color: var(--text-color) !important;
}

/* Target the up/down small buttons inside number inputs (some Streamlit versions) */
[data-testid="stAppViewContainer"] .stNumberInput button,
[data-testid="stSidebar"] .stNumberInput button {
  background: transparent !important;
  color: var(--text-color) !important;
}

/* Also make dropdown option text readable when the menu opens */
[role="listbox"] * { color: var(--text-color) !important; }

/* ------- Target selectbox buttons, opened option list, and radio groups ------- */

/* The visible selectbox control (the clickable button that shows current value) */
.stSelectbox > div,
.stSelectbox > div > button,
.stSelectbox > div[role="button"],
button[role="button"][aria-haspopup="listbox"],
div[role="button"][aria-haspopup="listbox"] {
  color: var(--text-color) !important;
  fill: var(--text-color) !important;
  background: var(--bg-secondary) !important;
  border: 1px solid rgba(0,0,0,0.06) !important;
}

/* Specific inner spans/divs inside the visible select control */
button[role="button"][aria-haspopup="listbox"] span,
button[role="button"][aria-haspopup="listbox"] div,
div[role="button"][aria-haspopup="listbox"] span,
div[role="button"][aria-haspopup="listbox"] div {
  color: var(--text-color) !important;
  background: transparent !important;
}

/* When the dropdown opens, options use role="option" (makes option text readable) */
[role="listbox"] { background: var(--bg-secondary) !important; color: var(--text-color) !important; }
[role="listbox"] [role="option"], [role="option"] {
  color: var(--text-color) !important;
  background: var(--bg-secondary) !important;
}
[role="option"][aria-selected="true"] { background: var(--bg-secondary) !important; color: var(--text-color) !important; }

/* Radio groups and radio options (prepay mode / tenure mode if rendered as radio) */
[role="radiogroup"], [role="radiogroup"] * , [role="radio"], [role="radio"] * {
  color: var(--text-color) !important;
  fill: var(--text-color) !important;
  background: transparent !important;
}

/* Also target label elements associated with inputs/selects (extra coverage) */
label, .css-1q8dd3e, .css-1okebmr, .css-1d391kg {
  color: var(--text-color) !important;
}

/* If the select control is wrapped in a div with aria-expanded attribute */
div[aria-expanded="false"], div[aria-expanded="true"] {
  color: var(--text-color) !important;
  background: var(--bg-secondary) !important;
}
"""
_DARK_THEME = """
:root {
  --bg-primary: #0f172a;
  --bg-secondary: #1e293b;
  --bg-card: #1e293b;
  --text-color: #e2e8f0;
  --text-muted: #94a3b8;
  --accent: #38bdf8;
  --positive: #22c55e;
  --negative: #ef4444;
}
[data-testid="stAppViewContainer"] { background: var(--bg-primary); color: var(--text-color); }
[data-testid="stSidebar"] { background: var(--bg-secondary); color: var(--text-color); }
.ui-card { background: var(--bg-card); border-radius: 12px; padding: 14px; box-shadow: 0 4px 12px rgba(0,0,0,0.6); margin-bottom: 12px; border: 1px solid rgba(255,255,255,0.08); }
.ui-muted { color: var(--text-muted) !important; }
.ui-metric-label { font-weight: 600; color: var(--text-muted); font-size: 0.95rem; }
.ui-metric-value { font-weight: 700; font-size: 1.4rem; color: var(--accent); }
.ui-badge { padding:4px 10px; border-radius:999px; font-weight:600; font-size:0.85rem; }
.ui-pos { background: rgba(34,197,94,0.15); color: var(--positive); }
.ui-neg { background: rgba(239,68,68,0.15); color: var(--negative); }

section.main > div.block-container { padding-top: 8px; padding-left: 18px; padding-right: 18px; }
.stDataFrame table, .stTable { background: var(--bg-secondary); color: var(--text-color); border-radius: 8px; }

/* enforce theme text colors for all common elements */
[data-testid="stAppViewContainer"] * { color: var(--text-color) !important; }
[data-testid="stSidebar"] * { color: var(--text-color) !important; }

/* links should use accent */
a, a:visited, a:active, .stMarkdown a { color: var(--accent) !important; }

/* muted captions should use text-muted */
.stCaption, .css-1lsmgbg, .stText, .stMarkdown { color: var(--text-muted) !important; }

/* table / dataframe text */
.stDataFrame table td, .stTable td, .stDataFrame table th, .stTable th { color: var(--text-color) !important; }

/* buttons/inputs labels fallback */
button, input, label, .stNumberInput, .stSelectbox { color: var(--text-color) !important; }

/* --- Widget / input specific fixes --- */
/* Number & text inputs (Streamlit widgets) */
[data-testid="stAppViewContainer"] input[type="number"],
[data-testid="stAppViewContainer"] input[type="text"],
[data-testid="stSidebar"] input[type="number"],
[data-testid="stSidebar"] input[type="text"],
.stNumberInput input,
.stTextInput input,
.stTextArea textarea {
  background: var(--bg-secondary) !important;    /* match theme background */
  color: var(--text-color) !important;           /* readable text color per theme */
  caret-color: var(--accent) !important;
  border-color: rgba(255,255,255,0.06) !important;
}

/* Placeholder / faint captions inside inputs */
[data-testid="stAppViewContainer"] input::placeholder,
[data-testid="stSidebar"] input::placeholder,
.stNumberInput input::placeholder,
.stTextInput input::placeholder {
  color: var(--text-muted) !important;
  opacity: 1 !important;
}

/* Select boxes and dropdown value text */
.stSelectbox > div, .stSelectbox button, .streamlit-expanderHeader {
  color: var(--text-color) !important;
}

/* Ensure table cells also have readable text & backgrounds */
.stDataFrame table td, .stTable td, .stDataFrame table th, .stTable th {
  background: transparent !important;
  color: var(--text-color) !important;
}

/* small: numeric spinner buttons keep native appearance but readable */
input[type="number"]::-webkit-inner-spin-button,
input[type="number"]::-webkit-outer-spin-button {
  -webkit-appearance: inner-spin-button;
}

/* --- Fix icons, SVGs and widget labels (plus/minus, dropdown arrows, radio/select/checkbox labels) --- */
/* Force SVG/icon fills and colors to follow theme text color */
[data-testid="stAppViewContainer"] svg,
[data-testid="stSidebar"] svg,
[data-testid="stAppViewContainer"] button svg,
[data-testid="stSidebar"] button svg {
  fill: var(--text-color) !important;
  color: var(--text-color) !important;
}

/* Make the label/text inside common widgets readable */
[data-testid="stAppViewContainer"] .stSelectbox *,
[data-testid="stAppViewContainer"] .stRadio *,
[data-testid="stAppViewContainer"] .stCheckbox *,
[data-testid="stAppViewContainer"] .stNumberInput *,
[data-testid="stAppViewContainer"] .stSlider *,
[data-testid="stSidebar"] .stSelectbox *,
[data-testid="stSidebar"] .stRadio *,
[data-testid="stSidebar"] .stCheckbox *,
[data-testid="stSidebar"] .stNumberInput * {
  color: var(--text-color) !important;
}

/* Target the up/down small buttons inside number inputs (some Streamlit versions) */
[data-testid="stAppViewContainer"] .stNumberInput button,
[data-testid="stSidebar"] .stNumberInput button {
  background: transparent !important;
  color: var(--text-color) !important;
}

/* Also make dropdown option text readable when the menu opens */
[role="listbox"] * { color: var(--text-color) !important; }

/* ------- Target selectbox buttons, opened option list, and radio groups ------- */

/* The visible selectbox control (the clickable button that shows current value) */
.stSelectbox > div,
.stSelectbox > div > button,
.stSelectbox > div[role="button"],
button[role="button"][aria-haspopup="listbox"],
div[role="button"][aria-haspopup="listbox"] {
  color: var(--text-color) !important;
  fill: var(--text-color) !important;
  background: var(--bg-secondary) !important;
  border: 1px solid rgba(255,255,255,0.06) !important;
}

/* Specific inner spans/divs inside the visible select control */
button[role="button"][aria-haspopup="listbox"] span,
button[role="button"][aria-haspopup="listbox"] div,
div[role="button"][aria-haspopup="listbox"] span,
div[role="button"][aria-haspopup="listbox"] div {
  color: var(--text-color) !important;
  background: transparent !important;
}

/* When the dropdown opens, options use role="option" (makes option text readable) */
[role="listbox"] { background: var(--bg-secondary) !important; color: var(--text-color) !important; }
[role="listbox"] [role="option"], [role="option"] {
  color: var(--text-color) !important;
  background: var(--bg-secondary) !important;
}
[role="option"][aria-selected="true"] { background: var(--bg-secondary) !important; color: var(--text-color) !important; }

/* Radio groups and radio options (prepay mode / tenure mode if rendered as radio) */
[role="radiogroup"], [role="radiogroup"] * , [role="radio"], [role="radio"] * {
  color: var(--text-color) !important;
  fill: var(--text-color) !important;
  background: transparent !important;
}

/* Also target label elements associated with inputs/selects (extra coverage) */
label, .css-1q8dd3e, .css-1okebmr, .css-1d391kg {
  color: var(--text-color) !important;
}

/* If the select control is wrapped in a div with aria-expanded attribute */
div[aria-expanded="false"], div[aria-expanded="true"] {
  color: var(--text-color) !important;
  background: var(--bg-secondary) !important;
}
"""

# -------------------------
# Theme selector (use selectbox for compatibility)
# -------------------------
with st.sidebar:
    theme_choice = st.selectbox("Theme", ["Light", "Dark"], index=0)

def _inject_theme(theme_css: str):
    st.markdown(f"<style>{theme_css}</style>", unsafe_allow_html=True)

if theme_choice == "Dark":
    _inject_theme(_DARK_THEME)
else:
    _inject_theme(_LIGHT_THEME)

# -------------------------
# Small UI helpers
# -------------------------
def ui_topbar(title: str, subtitle: str = "", right_text: str = ""):
    st.markdown(
        f"<div style='position:sticky;top:0;z-index:999;padding:10px 18px;background:transparent;display:flex;align-items:center;gap:12px;border-radius:0 0 10px 10px'><div><h1 style='margin:0;color:inherit'>{title}</h1><div style='color:var(--text-muted);font-size:12px'>{subtitle}</div></div><div style='margin-left:auto;color:var(--text-muted);font-weight:600'>{right_text}</div></div>",
        unsafe_allow_html=True,
    )


def ui_summary_cards(cards: list):
    st.markdown("<div style='display:flex;gap:12px;flex-wrap:wrap'>", unsafe_allow_html=True)
    for c in cards:
        label = c.get('label','')
        value = c.get('value','')
        hint = c.get('hint','')
        trend = c.get('trend', None)
        st.markdown("<div style='flex:1 1 220px;min-width:200px' class='ui-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='ui-metric-label'>{label}</div>", unsafe_allow_html=True)
        if trend == 'pos':
            st.markdown(f"<div style='display:flex;justify-content:space-between;align-items:center'><div class='ui-metric-value'>{value}</div><div class='ui-badge ui-pos'>Benefit</div></div>", unsafe_allow_html=True)
        elif trend == 'neg':
            st.markdown(f"<div style='display:flex;justify-content:space-between;align-items:center'><div class='ui-metric-value'>{value}</div><div class='ui-badge ui-neg'>Higher cost</div></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='ui-metric-value'>{value}</div>", unsafe_allow_html=True)
        if hint:
            st.markdown(f"<div class='ui-muted' style='margin-top:8px'>{hint}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def ui_info(text: str):
    st.markdown(f"<div class='ui-card'><div class='ui-muted'>{text}</div></div>", unsafe_allow_html=True)

# -------------------------
# Helpers (robust)
# -------------------------

def calculate_emi(P: float, annual_rate_percent: float, tenure_months: int) -> float:
    r = (annual_rate_percent / 100.0) / 12.0
    N = int(tenure_months)
    if N <= 0:
        return 0.0
    if abs(r) < 1e-12:
        return P / N
    pow_term = (1 + r) ** N
    emi = P * r * pow_term / (pow_term - 1)
    return emi


def months_for_fixed_emi(P: float, annual_rate_percent: float, emi: float) -> Optional[float]:
    r = (annual_rate_percent / 100.0) / 12.0
    if emi <= r * P + 1e-12:
        return None
    if abs(r) < 1e-12:
        return P / emi
    val = emi / (emi - r * P)
    if val <= 0:
        return None
    N = math.log(val) / math.log(1 + r)
    return N


def excel_download_bytes(df: pd.DataFrame, name="schedule"):
    out = BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Schedule")
    return out.getvalue()


def normalize_rate_schedule(rate_schedule_input: Optional[List[Dict]]) -> List[Dict]:
    if not rate_schedule_input:
        return []
    cleaned = []
    for ent in rate_schedule_input:
        if not isinstance(ent, dict):
            continue
        m = ent.get("month", None)
        r = ent.get("annual_rate", None)
        try:
            m_i = int(m)
            r_f = float(r)
            if m_i < 1:
                continue
            cleaned.append({"month": m_i, "annual_rate": r_f})
        except Exception:
            continue
    cleaned_sorted = sorted(cleaned, key=lambda x: x["month"])
    if not cleaned_sorted:
        return []
    if cleaned_sorted[0]["month"] != 1:
        first_rate = cleaned_sorted[0]["annual_rate"]
        cleaned_sorted.insert(0, {"month": 1, "annual_rate": float(first_rate)})
    return cleaned_sorted


def get_rate_for_month(rate_schedule: Optional[List[Dict]], month: int, fallback_rate: float) -> float:
    if not rate_schedule:
        return float(fallback_rate)
    applicable = None
    for ent in rate_schedule:
        try:
            if int(ent.get("month", 0)) <= month:
                applicable = ent
            else:
                break
        except Exception:
            continue
    return float(applicable["annual_rate"]) if applicable is not None else float(fallback_rate)

# -------------------------
# Core schedule builders (cached)
# -------------------------
@st.cache_data(show_spinner=False)
def build_schedule_with_prepay(
    P: float,
    annual_rate: float,
    tenure_months: int,
    prepay_amount: float = 0.0,
    prepay_month: Optional[int] = None,
    mode: str = "keep_emi",
    hybrid_frac: float = 0.0,
    rate_schedule: Optional[List[Dict]] = None,
    auto_recompute_on_rate_change: bool = False
) -> pd.DataFrame:
    rs = normalize_rate_schedule(rate_schedule) if rate_schedule is not None else []
    initial_rate = get_rate_for_month(rs, 1, annual_rate)
    emi = calculate_emi(P, initial_rate, tenure_months)
    balance = float(P)
    month = 0
    rows = []
    cum_interest = 0.0
    cum_principal = 0.0

    def recompute_emi_for_balance(rem_balance, rem_months, current_month_index):
        if rem_months <= 0:
            return rem_balance
        current_rate = get_rate_for_month(rs, current_month_index + 1, annual_rate)
        return calculate_emi(rem_balance, current_rate, rem_months)

    max_iter = max(int(tenure_months) * 10, 10000)
    rate_change_months = set([r["month"] for r in rs]) if rs else set()

    if emi <= 0:
        return pd.DataFrame(rows)

    while balance > 0.005 and month < max_iter:
        month += 1

        if auto_recompute_on_rate_change and (month in rate_change_months) and month != 1:
            remaining_months = max(1, int(tenure_months) - month + 1)
            emi = recompute_emi_for_balance(balance, remaining_months, month - 1)

        opening = balance
        curr_annual_rate = get_rate_for_month(rs, month, annual_rate)
        r = (curr_annual_rate / 100.0) / 12.0
        interest = opening * r
        principal_component = emi - interest

        if principal_component > opening:
            principal_component = opening
            emi_actual = interest + principal_component
        else:
            emi_actual = emi

        closing_before_prepay = opening - principal_component

        applied_prepay = 0.0
        applied_prepay_for_emi = 0.0
        applied_prepay_for_tenor = 0.0

        if prepay_amount > 0 and prepay_month is not None and prepay_month == month:
            if mode == "keep_emi":
                applied_prepay_for_tenor = min(prepay_amount, max(0.0, closing_before_prepay))
                applied_prepay = applied_prepay_for_tenor
            elif mode == "keep_tenure":
                applied_prepay_for_emi = min(prepay_amount, max(0.0, closing_before_prepay))
                applied_prepay = applied_prepay_for_emi
            elif mode == "hybrid":
                frac = float(hybrid_frac)
                frac = max(0.0, min(1.0, frac))
                amt_for_emi = prepay_amount * frac
                amt_for_tenor = prepay_amount * (1.0 - frac)
                applied_prepay_for_tenor = min(amt_for_tenor, max(0.0, closing_before_prepay))
                interim = closing_before_prepay - applied_prepay_for_tenor
                applied_prepay_for_emi = min(amt_for_emi, max(0.0, interim))
                applied_prepay = applied_prepay_for_tenor + applied_prepay_for_emi
            else:
                applied_prepay_for_tenor = min(prepay_amount, max(0.0, closing_before_prepay))
                applied_prepay = applied_prepay_for_tenor

        closing_after_prepay = max(closing_before_prepay - applied_prepay, 0.0)

        cum_interest += interest
        cum_principal += principal_component + applied_prepay

        rows.append({
            "Month": month,
            "Opening Balance": round(opening, 2),
            "EMI": round(emi_actual, 2),
            "Interest": round(interest, 2),
            "Principal Paid": round(principal_component, 2),
            "Prepay (Tenor part)": round(applied_prepay_for_tenor, 2),
            "Prepay (EMI part)": round(applied_prepay_for_emi, 2),
            "Prepay (Total)": round(applied_prepay, 2),
            "Closing Balance": round(closing_after_prepay, 2),
            "Applied Annual Rate (%)": round(curr_annual_rate, 6),
            "Cumulative Interest": round(cum_interest, 2),
            "Cumulative Principal Paid": round(cum_principal, 2)
        })

        balance = closing_after_prepay

        if applied_prepay > 0 and (mode == "keep_tenure" or (mode == "hybrid" and hybrid_frac > 0)):
            if balance > 0:
                remaining_months = max(1, int(tenure_months) - month)
                emi = recompute_emi_for_balance(balance, remaining_months, month)
            else:
                emi = 0.0

        next_rate = get_rate_for_month(rs, month + 1, annual_rate)
        next_r_monthly = (next_rate / 100.0) / 12.0
        if emi <= next_r_monthly * balance + 1e-12 and balance > 0.0:
            rows.append({
                "Month": month + 1,
                "Opening Balance": round(balance, 2),
                "EMI": round(emi, 2),
                "Interest": round(balance * next_r_monthly, 2),
                "Principal Paid": 0.0,
                "Prepay (Tenor part)": 0.0,
                "Prepay (EMI part)": 0.0,
                "Prepay (Total)": 0.0,
                "Closing Balance": round(balance + balance * next_r_monthly, 2),
                "Applied Annual Rate (%)": round(next_rate, 6),
                "Cumulative Interest": round(cum_interest + balance * next_r_monthly, 2),
                "Cumulative Principal Paid": round(cum_principal, 2)
            })
            break

        if balance <= 0.005:
            break

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def build_schedule_fixed_emi(P: float, annual_rate: float, fixed_emi: float, rate_schedule: Optional[List[Dict]] = None, max_months=1000) -> pd.DataFrame:
    rs = normalize_rate_schedule(rate_schedule) if rate_schedule is not None else []
    balance = float(P)
    month = 0
    rows = []
    cum_interest = 0.0
    cum_principal = 0.0
    while balance > 0.005 and month < max_months:
        month += 1
        opening = balance
        curr_annual_rate = get_rate_for_month(rs, month, annual_rate)
        r = (curr_annual_rate / 100.0) / 12.0
        interest = opening * r
        principal_component = fixed_emi - interest
        if principal_component <= 0:
            rows.append({
                "Month": month,
                "Opening Balance": round(opening, 2),
                "EMI": round(fixed_emi, 2),
                "Interest": round(interest, 2),
                "Principal Paid": 0.0,
                "Prepay (Total)": 0.0,
                "Closing Balance": round(opening + interest, 2),
                "Applied Annual Rate (%)": round(curr_annual_rate, 6),
                "Cumulative Interest": round(cum_interest + interest, 2),
                "Cumulative Principal Paid": round(cum_principal, 2)
            })
            cum_interest += interest
            balance = opening + interest
            break
        if principal_component > opening:
            principal_component = opening
            emi_actual = interest + principal_component
        else:
            emi_actual = fixed_emi
        closing = opening - principal_component
        cum_interest += interest
        cum_principal += principal_component
        rows.append({
            "Month": month,
            "Opening Balance": round(opening, 2),
            "EMI": round(emi_actual, 2),
            "Interest": round(interest, 2),
            "Principal Paid": round(principal_component, 2),
            "Prepay (Total)": 0.0,
            "Closing Balance": round(max(closing, 0.0), 2),
            "Applied Annual Rate (%)": round(curr_annual_rate, 6),
            "Cumulative Interest": round(cum_interest, 2),
            "Cumulative Principal Paid": round(cum_principal, 2)
        })
        balance = closing
        if balance <= 0.005:
            break
    return pd.DataFrame(rows)

# -------------------------
# Sidebar inputs (reorganized for clarity)
# -------------------------
with st.sidebar:
    st.markdown("## Loan inputs")

    principal = st.number_input("Principal (₹)", min_value=0.0, value=300000.0, step=500.0, format="%.2f", help="Total loan amount borrowed.")
    tenure_mode = st.selectbox("Tenure mode", ("Years", "Months"))
    if tenure_mode == "Years":
        years = st.number_input("Tenure (years)", min_value=0, value=20, step=1)
        tenure_months = int(years * 12)
    else:
        tenure_months = int(st.number_input("Tenure (months)", min_value=1, value=240, step=1))

    st.markdown("---")
    rate_type = st.selectbox("Interest-rate type", ("Fixed", "Flexible"), help="Choose fixed (one rate) or flexible (rates can change over time).")

    if rate_type == "Fixed":
        annual_rate = st.number_input("Annual interest rate (%)", min_value=0.0, value=8.0, step=0.01, format="%.3f", help="Annual fixed rate (in %).")
        rate_schedule = None
        st.caption("Fixed rate: same annual rate for the entire tenure.")
    else:
        st.markdown("Flexible rates — enter initial rate and optional changes.")
        initial_rate = st.number_input("Initial annual rate (%) (month 1)", min_value=0.0, value=8.0, step=0.01, format="%.3f")
        num_changes = int(st.number_input("Number of scheduled rate changes (besides simulator)", min_value=0, value=0, step=1))
        changes = []
        for i in range(int(num_changes)):
            st.markdown(f"Scheduled change #{i+1}")
            ch_month = int(st.number_input(f"Change #{i+1}: month (effective)", min_value=2, max_value=tenure_months, value=min(13 + i*6, tenure_months), key=f"cm{i}"))
            ch_rate = float(st.number_input(f"Change #{i+1}: annual rate (%)", min_value=0.0, value=9.0 + i*0.5, step=0.01, format="%.3f", key=f"cr{i}"))
            changes.append({"month": ch_month, "annual_rate": ch_rate})
        rate_schedule = normalize_rate_schedule([{"month": 1, "annual_rate": float(initial_rate)}] + changes)
        annual_rate = float(initial_rate)
        st.caption("Flexible: schedule planned changes. Use Simulator below to model bank shocks.")

    st.markdown("---")
    enable_prepay = st.checkbox("Enable one-time prepayment", value=False, help="Apply a one-time prepayment after the chosen month. Prepayment is applied after that month's EMI.")
    if enable_prepay:
        prepay_amount = st.number_input("Prepayment amount (₹)", min_value=0.0, value=100000.0, step=100.0, format="%.2f")
        prepay_month = int(st.number_input("Apply after month #", min_value=1, value=14, step=1))
        prepay_mode = st.selectbox("Prepay mode", ("Keep EMI (shorten tenure)", "Keep Tenure (reduce EMI)", "Hybrid (split)"))
        if prepay_mode == "Hybrid (split)":
            hybrid_pct = st.slider("Hybrid: % to reduce EMI (rest shortens tenure)", min_value=0, max_value=100, value=50)
            hybrid_frac = hybrid_pct / 100.0
        else:
            hybrid_frac = 0.0
    else:
        prepay_amount = 0.0
        prepay_month = None
        prepay_mode = None
        hybrid_frac = 0.0

    # Simulator only visible for Flexible
    sim_enabled = False
    sim_month = None
    sim_new_rate = None
    if rate_type == "Flexible":
        st.markdown("---")
        st.write("Bank Rate-Change Simulator (Flexible only)")
        st.caption("Model a bank deciding to change the benchmark rate at a chosen month. This will override subsequent scheduled rates.")
        sim_enabled = st.checkbox("Enable bank rate-change simulator", value=False)
        if sim_enabled:
            sim_month = int(st.number_input("Month when bank raises rate (simulator)", min_value=1, max_value=tenure_months, value=60))
            sim_new_rate = float(st.number_input("New annual rate (%) from sim-month onward", min_value=0.0, value=11.0, step=0.01, format="%.3f"))
            st.caption("When enabled, the app inserts this change at sim-month and re-runs amortization with EMI recomputed at every rate change.")

    st.markdown("---")
    st.markdown("<div style='color:var(--text-muted);font-size:12px'>Tip: Use the Visuals tab to inspect charts. Theme control above toggles Light/Dark mode.</div>", unsafe_allow_html=True)

# -------------------------
# Validate inputs
# -------------------------
if principal <= 0:
    st.sidebar.error("Principal must be greater than 0.")
    st.stop()

if tenure_months <= 0:
    st.sidebar.error("Tenure must be at least 1 month.")
    st.stop()

if enable_prepay and prepay_month is not None and prepay_month > tenure_months:
    st.sidebar.error("Prepayment month cannot be after loan tenure.")
    st.stop()

# -------------------------
# Prepare rate schedule variable for use
# -------------------------
rs_input = rate_schedule  # explicit and simple

# -------------------------
# Compute schedules (wrapped in try/except to show errors gracefully)
# -------------------------
try:
    df_base = build_schedule_with_prepay(
        P=principal, annual_rate=annual_rate, tenure_months=tenure_months,
        prepay_amount=0.0, prepay_month=None, mode="keep_emi", hybrid_frac=0.0,
        rate_schedule=rs_input, auto_recompute_on_rate_change=False
    )
except Exception as e:
    st.error(f"Failed to build base schedule: {e}")
    df_base = pd.DataFrame()


df_prepay = None
if enable_prepay:
    try:
        mode_key = {"Keep EMI (shorten tenure)": "keep_emi", "Keep Tenure (reduce EMI)": "keep_tenure", "Hybrid (split)": "hybrid"}[prepay_mode]
    except Exception:
        mode_key = "keep_emi"
    try:
        df_prepay = build_schedule_with_prepay(
            P=principal, annual_rate=annual_rate, tenure_months=tenure_months,
            prepay_amount=prepay_amount, prepay_month=prepay_month, mode=mode_key, hybrid_frac=hybrid_frac,
            rate_schedule=rs_input, auto_recompute_on_rate_change=False
        )
    except Exception as e:
        st.error(f"Failed to build prepay schedule: {e}")
        df_prepay = pd.DataFrame()

# Simulator: create simulated schedule only when Flexible & sim_enabled

df_sim_base = None
df_sim_prepay = None
sim_rs = []
if rate_type == "Flexible" and sim_enabled and sim_month is not None and sim_new_rate is not None:
    try:
        base_rs = rs_input or [{"month": 1, "annual_rate": annual_rate}]
        rs_before = [r for r in base_rs if r["month"] < sim_month]
        sim_rs = rs_before + [{"month": sim_month, "annual_rate": float(sim_new_rate)}]
        sim_rs = normalize_rate_schedule(sim_rs)

        df_sim_base = build_schedule_with_prepay(
            P=principal, annual_rate=annual_rate, tenure_months=tenure_months,
            prepay_amount=0.0, prepay_month=None, mode="keep_emi", hybrid_frac=0.0,
            rate_schedule=sim_rs, auto_recompute_on_rate_change=True
        )

        if enable_prepay:
            df_sim_prepay = build_schedule_with_prepay(
                P=principal, annual_rate=annual_rate, tenure_months=tenure_months,
                prepay_amount=prepay_amount, prepay_month=prepay_month, mode=mode_key, hybrid_frac=hybrid_frac,
                rate_schedule=sim_rs, auto_recompute_on_rate_change=True
            )
    except Exception as e:
        st.error(f"Failed to run simulator: {e}")
        df_sim_base = pd.DataFrame()
        df_sim_prepay = pd.DataFrame()

# -------------------------
# Analysis: interest saved by prepayment and effect of bank change
# -------------------------

def compute_summary(df):
    if df is None or df.empty:
        return {"months": 0, "total_interest": 0.0, "total_principal": 0.0, "total_paid": 0.0}
    months = int(df["Month"].max())
    total_interest = float(df["Interest"].sum())
    total_principal = float(df["Principal Paid"].sum() + df.get("Prepay (Total)", pd.Series(dtype=float)).sum())
    total_paid = float(df["EMI"].sum() + df.get("Prepay (Total)", pd.Series(dtype=float)).sum())
    return {"months": months, "total_interest": total_interest, "total_principal": total_principal, "total_paid": total_paid}

base_s = compute_summary(df_base)
pre_s = compute_summary(df_prepay) if df_prepay is not None else None
sim_s = compute_summary(df_sim_base) if df_sim_base is not None else None
sim_pre_s = compute_summary(df_sim_prepay) if df_sim_prepay is not None else None

# Interest saved by prepayment (original schedule)
interest_saved = None
if df_prepay is not None and not df_prepay.empty:
    interest_saved = base_s["total_interest"] - pre_s["total_interest"]

# Effect of bank change: more or less interest for consumer?
sim_interest_delta = None
if df_sim_base is not None and not df_sim_base.empty:
    sim_interest_delta = sim_s["total_interest"] - base_s["total_interest"]

# When prepay exists, compute whether prepay still saves interest under simulated scenario
sim_pre_saved = None
if df_sim_prepay is not None and not df_sim_prepay.empty and df_sim_base is not None and not df_sim_base.empty:
    sim_pre_saved = sim_s["total_interest"] - sim_pre_s["total_interest"]

# -------------------------
# Render UI — topbar + tabs
# -------------------------
ui_topbar('Loan EMI Planner — Rates & Prepayment Analysis', 'Compare fixed vs flexible loans, run a bank-rate simulator, and model prepayments.', 'v1.3')

tabs = st.tabs(["Overview", "Visuals", "Schedules & Export", "Diagnostics"])

# ---------- Overview tab ----------
with tabs[0]:
    st.header("Summary — Key metrics")
    cards = [
        {"label":"Original months", "value": f"{base_s['months']}", "hint":"Loan length (months)"},
        {"label":"Original total interest", "value": f"₹{base_s['total_interest']:,.0f}", "hint":"Interest paid without prepayment"},
        {"label":"Interest saved by prepayment", "value": f"₹{interest_saved:,.0f}" if interest_saved is not None else "—", "hint":"Lower is better", "trend":"pos" if interest_saved and interest_saved>0 else None},
        {"label":"Bank change effect", "value": f"₹{sim_interest_delta:,.0f}" if sim_interest_delta is not None else "—", "hint":"Positive = you pay more", "trend":"neg" if sim_interest_delta and sim_interest_delta>0 else ("pos" if sim_interest_delta and sim_interest_delta<0 else None)}
    ]
    ui_summary_cards(cards)

    st.markdown("---")
    ui_info("Prepayment reduces outstanding principal immediately — earlier prepayments save more interest. Use the Visuals tab for monthly breakdowns and breakeven charts.")

# ---------- Visuals tab ----------
with tabs[1]:
    st.header("Balance & EMI split visuals")

    fig = go.Figure()
    if df_base is not None and not df_base.empty:
        fig.add_trace(go.Scatter(x=df_base["Month"], y=df_base["Opening Balance"], mode="lines", name="Base (original)"))
    if df_sim_base is not None and not df_sim_base.empty:
        fig.add_trace(go.Scatter(x=df_sim_base["Month"], y=df_sim_base["Opening Balance"], mode="lines", name="Simulated (bank change)"))
    if df_prepay is not None and not df_prepay.empty:
        fig.add_trace(go.Scatter(x=df_prepay["Month"], y=df_prepay["Opening Balance"], mode="lines", name="Base + Prepay"))
    if df_sim_prepay is not None and not df_sim_prepay.empty:
        fig.add_trace(go.Scatter(x=df_sim_prepay["Month"], y=df_sim_prepay["Opening Balance"], mode="lines", name="Simulated + Prepay"))
    fig.update_layout(title="Outstanding Opening Balance over time", xaxis_title="Month", yaxis_title="Balance (₹)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("EMI split (interest vs principal) — first N months")
    if df_base is not None and not df_base.empty:
        max_month_plot = int(df_base["Month"].max())
        show_n = st.slider("Show first N months", min_value=1, max_value=max_month_plot, value=min(24, max_month_plot))
        cols_plot = st.columns(2)
        with cols_plot[0]:
            st.markdown("Base (original)")
            small = df_base[df_base["Month"] <= show_n][["Month", "Interest", "Principal Paid"]]
            stacked = small.melt(id_vars="Month", value_vars=["Interest", "Principal Paid"], var_name="Type", value_name="Amount")
            figb = px.bar(stacked, x="Month", y="Amount", color="Type", barmode="stack", title=f"Base first {show_n} months")
            st.plotly_chart(figb, use_container_width=True)
        with cols_plot[1]:
            if df_sim_base is not None and not df_sim_base.empty:
                st.markdown("Simulated (after bank change)")
                small_s = df_sim_base[df_sim_base["Month"] <= show_n][["Month", "Interest", "Principal Paid"]]
                stacked_s = small_s.melt(id_vars="Month", value_vars=["Interest", "Principal Paid"], var_name="Type", value_name="Amount")
                figs = px.bar(stacked_s, x="Month", y="Amount", color="Type", barmode="stack", title=f"Simulated first {show_n} months")
                st.plotly_chart(figs, use_container_width=True)
            else:
                st.info("Simulator disabled or not applicable.")

    st.markdown("---")
    # Show breakeven chart if prepay enabled
    if enable_prepay and df_prepay is not None and not df_prepay.empty:
        st.subheader("Prepayment breakeven: cumulative interest saved")
        max_m = int(max(df_base["Month"].max(), df_prepay["Month"].max()))
        rows = []
        cum_base = cum_prep = 0.0
        breakeven_month = None
        for m in range(1, max_m + 1):
            ia = float(df_base[df_base["Month"] == m]["Interest"].iloc[0]) if not df_base[df_base["Month"] == m].empty else 0.0
            ib = float(df_prepay[df_prepay["Month"] == m]["Interest"].iloc[0]) if not df_prepay[df_prepay["Month"] == m].empty else 0.0
            cum_base += ia
            cum_prep += ib
            diff = cum_base - cum_prep
            rows.append({"Month": m, "Cumulative Interest Saved": diff})
            if breakeven_month is None and diff >= prepay_amount:
                breakeven_month = m

        df_rows = pd.DataFrame(rows)
        fig_bk = px.line(df_rows, x="Month", y="Cumulative Interest Saved", title="Cumulative interest saved (Base - Prepay)")
        fig_bk.add_hline(y=prepay_amount, line_dash="dash", annotation_text=f"Prepay amount ₹{prepay_amount:,.0f}", annotation_position="top left")
        st.plotly_chart(fig_bk, use_container_width=True)
        if breakeven_month is None:
            st.warning("Breakeven not reached within displayed term.")
        else:
            st.success(f"Breakeven at month {breakeven_month} (~{breakeven_month//12} years and {breakeven_month%12} months).")

# ---------- Schedules & Export tab ----------
with tabs[2]:
    st.header("Schedules & Export")
    st.markdown("Download the amortization schedules (Excel) for offline analysis.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("Base schedule (original)")
        if df_base is not None and not df_base.empty:
            st.dataframe(df_base.round(2), height=400)
            st.download_button("Download base schedule (.xlsx)", excel_download_bytes(df_base, "base"), "base_schedule.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.info("Base schedule not available.")
    with col_b:
        if df_sim_base is not None and not df_sim_base.empty:
            st.markdown("Simulated schedule (bank change)")
            st.dataframe(df_sim_base.round(2), height=400)
            st.download_button("Download simulated schedule (.xlsx)", excel_download_bytes(df_sim_base, "sim_base"), "sim_base_schedule.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.info("Simulated schedule not available or simulator disabled.")

    if enable_prepay and df_prepay is not None and not df_prepay.empty:
        st.markdown("---")
        st.subheader("Prepayment schedules & downloads")
        lcol, rcol = st.columns(2)
        with lcol:
            st.markdown("Base + Prepay")
            st.dataframe(df_prepay.round(2), height=320)
            st.download_button("Download base+prepay (.xlsx)", excel_download_bytes(df_prepay, "base_prepay"), "base_prepay_schedule.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with rcol:
            if df_sim_prepay is not None and not df_sim_prepay.empty:
                st.markdown("Simulated + Prepay")
                st.dataframe(df_sim_prepay.round(2), height=320)
                st.download_button("Download sim+prepay (.xlsx)", excel_download_bytes(df_sim_prepay, "sim_prepay"), "sim_prepay_schedule.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                st.info("Simulated prepay schedule not available.")

# ---------- Diagnostics tab ----------
with tabs[3]:
    st.header("Diagnostics & Notes")
    st.write("This tab helps debugging and understanding edge cases.")

    def detect_negative_amortization(df):
        if df is None or df.empty:
            return False
        neg = df[(df["Principal Paid"] <= 0.0) & (df["Closing Balance"] >= df["Opening Balance"]) ]
        return not neg.empty

    if detect_negative_amortization(df_base):
        st.error("Negative-amortization detected in base schedule: EMI does not cover monthly interest at some point.")
    else:
        st.success("No negative-amortization detected in base schedule.")

    if df_sim_base is not None and not df_sim_base.empty:
        if detect_negative_amortization(df_sim_base):
            st.error("Negative-amortization detected in simulated schedule.")
        else:
            st.success("No negative-amortization detected in simulated schedule.")

    st.markdown("**Rate schedules**")
    if rs_input:
        st.write("User-provided flexible schedule:")
        st.table(pd.DataFrame(rs_input))
    else:
        st.write(f"Fixed rate: {annual_rate:.3f}% (no schedule)")

    if sim_rs:
        st.write("Simulator-provided override schedule:")
        st.table(pd.DataFrame(sorted(sim_rs, key=lambda x: x['month'])))

    st.markdown("---")
    st.write(
        "- Prepayment is applied after the EMI of the chosen month.\n"
        "- Hybrid prepayment splits the prepay amount into a portion that reduces EMI (keep_tenure) and a portion that shortens tenure (keep_emi).\n"
        "- Simulator inserts a new rate entry at the chosen month and overrides later scheduled entries (so the bank action applies from that month onward)."
    )

st.markdown("<div style='text-align:center;color:var(--text-muted);margin-top:14px'>Made with care — Loan EMI Planner · UI refreshed (v1.3)</div>", unsafe_allow_html=True)

# End of file
