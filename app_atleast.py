# ════════════════════════════════════════════════════════════════════
#  ILALA DISTRICT ELECTRICAL LOAD FORECASTING SYSTEM
#  Streamlit Deployment App  (v5 — Long-Term / Dark Blue Theme)
#  Author: Jackline Boniphace — DIT M.Eng. Dissertation 2026
#
#  v5 CHANGES vs v4:
#   - Target corrected to Total_District_MW (Ilala + Gongo la Mboto +
#     Kipawa + NCC substations SUMMED), not the Ilala substation alone
#   - Neighbouring-substation input section REMOVED (those 4 raw
#     readings sum to the target, so using them as inputs would be
#     data leakage)
#   - NEW long-term factors added: Population, Households,
#     Total_Customers, Service_Line_Connections, Cumulative_Connections,
#     T1_LUKU, T2, T3, Year_index
#   - NEW 1-year load-memory features: Total_District_MW_lag8760h,
#     Total_District_MW_roll8760h_mean
#   - Forecast horizon reframed from short-term (1-168h) to long-term
#     (growth-aware, multi-year-capable)
# ════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import io
from datetime import datetime
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ── Page configuration ───────────────────────────────────────────────
st.set_page_config(
    page_title = "Ilala Long-Term Load Forecasting System",
    page_icon  = "⚡",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ════════════════════════════════════════════════════════════════════
#  GLOBAL STYLES — DARK BLUE THEME
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* Main background — dark blue */
    .stApp {
        background: linear-gradient(160deg, #0A1F3D 0%, #0E2A52 50%, #0A1F3D 100%);
    }

    /* Sidebar — even darker blue */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #061530 0%, #0A2145 100%);
        border-right: 1px solid #1F4E8C;
    }
    [data-testid="stSidebar"] * { color: #FFFFFF !important; }

    /* All headings — WHITE so they are visible on dark blue */
    h1, h2, h3, h4, h5 { color: #FFFFFF !important; }

    /* General text on the page */
    .stApp p, .stApp label, .stApp span, .stMarkdown { color: #E8EEF7; }

    /* Cards — dark navy with light text */
    .info-card {
        background: #122E5C;
        padding: 1.5rem;
        border-radius: 14px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.35);
        margin-bottom: 1rem;
        border-left: 5px solid #4DA3FF;
    }
    .info-card p, .info-card li, .info-card span, .info-card ol, .info-card ul {
        color: #E8EEF7 !important;
    }
    .metric-card {
        background: #122E5C;
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 3px 12px rgba(0,0,0,0.35);
        border-top: 4px solid #4DA3FF;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #FFFFFF; }
    .metric-label { font-size: 0.85rem; color: #9DB8DD; margin-top: 0.3rem; }

    /* Section headers — bright blue banner, white text */
    .section-header {
        background: linear-gradient(90deg, #1F4E8C, #2E86C1);
        color: #FFFFFF !important;
        padding: 0.8rem 1.5rem;
        border-radius: 12px;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 1rem;
        box-shadow: 0 3px 12px rgba(0,0,0,0.35);
    }

    /* Group titles inside manual-input form */
    .group-title {
        background: #1F4E8C;
        color: #FFFFFF;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.95rem;
        margin: 0.8rem 0 0.5rem 0;
        border-left: 4px solid #4DA3FF;
    }

    .success-box {
        background: #0F3D2E; border: 1px solid #2ECC71;
        border-radius: 10px; padding: 1rem; color: #7DE8AF;
    }

    /* Inputs — dark, readable */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb] {
        background-color: #0E2A52 !important;
        color: #FFFFFF !important;
    }
    .stSlider label, .stNumberInput label, .stSelectbox label,
    .stTextInput label, .stFileUploader label { color: #E8EEF7 !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab"] { color: #E8EEF7; }
    .stTabs [aria-selected="true"] { color: #4DA3FF !important; }

    /* DataFrames */
    [data-testid="stDataFrame"] { background: #122E5C; }

    /* Buttons — bright blue so they stand out */
    .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
        background: linear-gradient(90deg, #2E86C1, #4DA3FF);
        color: #FFFFFF; border: none; border-radius: 10px;
        padding: 0.6rem 2rem; font-weight: 700; width: 100%;
    }
    .stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ════════════════════════════════════════════════════════════════════
for key, default in [
    ('authenticated', False), ('current_page', 'About'),
    ('user_data', {}), ('uploaded_df', None), ('predictions', None),
    ('future_predictions', None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

CREDENTIALS = {'Jackline': 'jackline@2026'}

# Base year used for Year_index during training (min year remaining after
# the first calendar year, 2018, was dropped for a valid 1-year lag).
BASE_YEAR = 2019

# Socio-economic long-term driver columns (also used to fit growth trends
# for the future-year forecasting mode below).
SOCIO_COLS = ['Population', 'Households', 'Total_Customers',
              'Service_Line_Connections', 'Cumulative_Connections',
              'T1_LUKU', 'T2', 'T3']

# Fixed-date Tanzanian public holidays (month, day). Movable holidays that
# follow the Islamic lunar calendar (Eid al-Fitr, Eid al-Adha, Maulid) and
# Easter-linked dates (Good Friday, Easter Monday) are not included since
# they can't be projected onto arbitrary future years without a calendar
# conversion — treat this as an approximation for future-year forecasts.
TZ_HOLIDAYS_MMDD = {
    (1, 1): "New Year's Day", (1, 12): "Zanzibar Revolution Day",
    (4, 26): "Union Day", (5, 1): "Labour Day", (7, 7): "Saba Saba",
    (8, 8): "Nane Nane", (10, 14): "Nyerere Day",
    (12, 9): "Independence Day", (12, 25): "Christmas Day",
    (12, 26): "Boxing Day",
}

def get_season(month):
    """Same Season encoding used in the training data / manual-input form."""
    if month in (12, 1, 2):
        return 1  # Hot & Wet
    if month in (3, 4, 5):
        return 2  # Long Rains
    if month in (6, 7, 8):
        return 3  # Cool & Dry
    return 4      # Short Rains (9,10,11)

# ════════════════════════════════════════════════════════════════════
#  MODEL LOADER
# ════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model_assets():
    base = 'best_model_deployment'
    assets = {}
    meta_path = os.path.join(base, 'metadata.json')
    assets['metadata'] = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            assets['metadata'] = json.load(f)
    for fname in ['best_model.pkl', 'best_model.keras']:
        path = os.path.join(base, fname)
        if os.path.exists(path):
            if fname.endswith('.pkl'):
                assets['model'] = joblib.load(path)
            else:
                from tensorflow.keras.models import load_model
                assets['model'] = load_model(path)
            assets['model_file'] = fname
            break
    for key, fname in [('scaler_X', 'scaler_X.pkl'), ('scaler_y', 'scaler_y.pkl')]:
        path = os.path.join(base, fname)
        if os.path.exists(path):
            assets[key] = joblib.load(path)
    feat_path = os.path.join(base, 'feature_columns.pkl')
    if os.path.exists(feat_path):
        assets['feature_cols'] = joblib.load(feat_path)
    return assets

# ════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════
def make_prediction(model, X, metadata, scaler_y=None, scaler_X=None):
    model_name = metadata.get('best_model', '') if metadata else ''
    if 'LSTM' in model_name:
        timesteps = metadata.get('timesteps_lstm', 24)
        X_sc = scaler_X.transform(X)
        sequences = [X_sc[i - timesteps:i] for i in range(timesteps, len(X_sc))]
        if not sequences:
            return np.array([])
        y_sc = model.predict(np.array(sequences), verbose=0).ravel()
        return scaler_y.inverse_transform(y_sc.reshape(-1, 1)).ravel()
    return model.predict(X)

def compute_metrics(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    r2   = r2_score(y_true, y_pred)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}

def to_excel_bytes(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    return buf.getvalue()

def run_long_term_forecast(df_hist, target_year, feat_cols, model, metadata,
                            scaler_X, scaler_y, progress_cb=None):
    """
    Recursively forecasts Total_District_MW hour-by-hour from the end of the
    uploaded historical data through 31 Dec of `target_year`.

    Because the future actual load is unknown, lag/rolling features are
    computed strictly from prior hours (own predictions once past the
    historical tail) — unlike the upload-based 'nowcast' tab, which can use
    the same-hour actual value already present in an uploaded historical file.

    Weather is projected using each (month, hour) climatological average
    from the uploaded history. Socio-economic drivers (population,
    customers, connections, etc.) are extrapolated with a simple per-column
    linear trend fit on historical yearly averages. Holidays beyond the
    uploaded data use a fixed-date approximation (see TZ_HOLIDAYS_MMDD).
    """
    df_hist = df_hist.sort_values('Datetime').reset_index(drop=True)
    last_dt = df_hist['Datetime'].max()
    future_start = last_dt + pd.Timedelta(hours=1)
    future_end = pd.Timestamp(year=int(target_year), month=12, day=31, hour=23)
    if future_end <= last_dt:
        raise ValueError("Target year must be after the last date in your "
                          f"uploaded data ({last_dt.date()}).")

    future_index = pd.date_range(future_start, future_end, freq='h')
    n_future = len(future_index)

    hist_target = df_hist['Total_District_MW'].to_numpy(dtype=float)
    hist_len = len(hist_target)
    combined = np.concatenate([hist_target, np.full(n_future, np.nan)])

    # Weather climatology: mean per (month, hour) across the whole upload.
    clim = (df_hist.assign(_m=df_hist['Datetime'].dt.month,
                            _h=df_hist['Datetime'].dt.hour)
                   .groupby(['_m', '_h'])[['Temperature_C', 'Humidity_pct',
                                            'Rainfall_mm']]
                   .mean())
    overall_weather = df_hist[['Temperature_C', 'Humidity_pct',
                                'Rainfall_mm']].mean()

    # Linear growth trend per socio-economic column, fit on yearly averages.
    yearly = (df_hist.assign(_y=df_hist['Datetime'].dt.year)
                     .groupby('_y')[SOCIO_COLS].mean())
    trend_coefs = {}
    for col in SOCIO_COLS:
        if col in yearly.columns and yearly[col].notna().sum() >= 2:
            x = yearly.index.values.astype(float)
            y = yearly[col].values.astype(float)
            trend_coefs[col] = np.polyfit(x, y, 1)
        else:
            trend_coefs[col] = (0.0, df_hist[col].dropna().iloc[-1]
                                 if col in df_hist.columns else 0.0)

    out_rows = []
    update_every = max(1, n_future // 100)

    for i, dt in enumerate(future_index):
        t = hist_len + i
        hour, dow, month = dt.hour, dt.weekday(), dt.month
        doy, year = dt.timetuple().tm_yday, dt.year
        season = get_season(month)
        is_weekend = 1 if dow >= 5 else 0
        is_holiday = 1 if (month, dt.day) in TZ_HOLIDAYS_MMDD else 0
        is_workday = int(not is_weekend and not is_holiday)

        hour_sin, hour_cos = np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)
        month_sin, month_cos = np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12)
        dow_sin, dow_cos = np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7)
        doy_sin, doy_cos = np.sin(2*np.pi*doy/365), np.cos(2*np.pi*doy/365)
        year_index = year - BASE_YEAR

        try:
            temp, hum, rain = clim.loc[(month, hour)]
        except KeyError:
            temp, hum, rain = overall_weather
        temp_x_hour = temp * hour_sin

        def safe(idx):
            return combined[idx] if idx >= 0 else np.nan

        lag1, lag2, lag3 = safe(t-1), safe(t-2), safe(t-3)
        lag24, lag48, lag168 = safe(t-24), safe(t-48), safe(t-168)
        lag8760 = safe(t-8760)
        win = lambda k: combined[max(0, t-k):t]
        roll3  = np.nanmean(win(3))
        roll6  = np.nanmean(win(6))
        roll24 = np.nanmean(win(24))
        roll24std = np.nanstd(win(24), ddof=1) if t >= 2 else 0.0
        roll8760 = np.nanmean(win(8760))

        frac_year = year + (doy - 1) / 365.0
        socio_vals = {c: trend_coefs[c][0]*frac_year + trend_coefs[c][1]
                      for c in SOCIO_COLS}

        feat_row = {
            'Hour_of_Day': hour, 'Day_of_Week': dow, 'Month': month,
            'Season': season, 'Is_Weekend': is_weekend,
            'Is_Holiday': is_holiday, 'Is_Workday': is_workday,
            'Hour_sin': hour_sin, 'Hour_cos': hour_cos,
            'Month_sin': month_sin, 'Month_cos': month_cos,
            'DOW_sin': dow_sin, 'DOW_cos': dow_cos,
            'DOY_sin': doy_sin, 'DOY_cos': doy_cos,
            'Year_index': year_index,
            'Temperature_C': temp, 'Humidity_pct': hum, 'Rainfall_mm': rain,
            'Temp_x_Hour': temp_x_hour,
            'Total_District_MW_lag1h': lag1, 'Total_District_MW_lag2h': lag2,
            'Total_District_MW_lag3h': lag3, 'Total_District_MW_lag24h': lag24,
            'Total_District_MW_lag48h': lag48,
            'Total_District_MW_lag168h': lag168,
            'Total_District_MW_roll3h_mean': roll3,
            'Total_District_MW_roll6h_mean': roll6,
            'Total_District_MW_roll24h_mean': roll24,
            'Total_District_MW_roll24h_std': roll24std,
            'Total_District_MW_lag8760h': lag8760,
            'Total_District_MW_roll8760h_mean': roll8760,
            **socio_vals,
        }

        X_row = np.nan_to_num(
            np.array([[feat_row.get(c, 0.0) for c in feat_cols]], dtype=float),
            nan=0.0)
        pred = float(make_prediction(model, X_row, metadata,
                                      scaler_y, scaler_X)[0])
        combined[t] = pred

        row_out = {'Datetime': dt, 'Predicted_Total_District_MW': round(pred, 3)}
        row_out.update({k: v for k, v in feat_row.items()
                         if k in ('Temperature_C', 'Humidity_pct', 'Rainfall_mm',
                                  'Is_Weekend', 'Is_Holiday', 'Season')})
        out_rows.append(row_out)

        if progress_cb and (i % update_every == 0 or i == n_future - 1):
            progress_cb((i + 1) / n_future)

    return pd.DataFrame(out_rows)

# Shared dark-theme layout for all Plotly charts
DARK_CHART = dict(
    plot_bgcolor  = '#122E5C',
    paper_bgcolor = '#122E5C',
    font          = dict(color='#E8EEF7', family='Arial', size=12),
)

def dark_axes(fig):
    fig.update_xaxes(showgrid=True, gridcolor='#1F4E8C',
                     zerolinecolor='#1F4E8C', color='#E8EEF7')
    fig.update_yaxes(showgrid=True, gridcolor='#1F4E8C',
                     zerolinecolor='#1F4E8C', color='#E8EEF7')
    return fig

# ════════════════════════════════════════════════════════════════════
#  PAGE — LOGIN  (only shown BEFORE authentication)
# ════════════════════════════════════════════════════════════════════
def page_login():
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align:center; margin-bottom:2rem;'>
            <div style='font-size:3.5rem;'>⚡</div>
            <h2 style='color:#FFFFFF; margin:0;'>Ilala Load Forecasting</h2>
            <p style='color:#9DB8DD; font-size:0.9rem;'>
                Long-Term Electrical Load Forecasting System<br>
                Ilala District, Dar es Salaam</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🔐 System Login")
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password",
                                 placeholder="Enter your password")
        if st.button("Login  →"):
            if username in CREDENTIALS and CREDENTIALS[username] == password:
                st.session_state.authenticated = True
                st.session_state.username      = username
                st.session_state.current_page  = 'About'
                st.rerun()
            else:
                st.error("Incorrect username or password. Please try again.")

        st.markdown("""
        <p style='text-align:center; color:#5D7CA6; font-size:0.78rem;
                  margin-top:1.5rem;'>
            M.Eng. Sustainable Energy Dissertation — DIT 2026<br>
            Jackline Boniphace</p>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
#  PAGE — ABOUT
# ════════════════════════════════════════════════════════════════════
def page_about():
    st.markdown('<div class="section-header">⚡ About the System</div>',
                unsafe_allow_html=True)
    col1, col2 = st.columns([1.6, 1])

    with col1:
        st.markdown("""
        <div class="info-card">
            <h3 style='margin-top:0;'>What This System Does</h3>
            <p style='line-height:1.8;'>
            This system forecasts <strong>long-term</strong> electricity
            demand growth for <strong>Ilala District, Dar es Salaam</strong>
            using machine learning. The target is the district's
            <strong>total load</strong> — the Ilala, Gongo la Mboto, Kipawa
            and NCC substations combined — driven not only by recent load
            history but by long-term structural growth factors: population,
            households, customer connections and electrification drive.
            This supports TANESCO planners in capacity expansion and
            generation-investment decisions, not just next-hour dispatch.</p>
        </div>
        <div class="info-card">
            <h3 style='margin-top:0;'>How to Use the System</h3>
            <ol style='line-height:2;'>
                <li>Go to <strong>Registration</strong> — register your details
                    and upload your load data file (CSV / Excel / JSON)</li>
                <li>Go to <strong>Prediction</strong> — run a forecast from your
                    uploaded data, or enter values manually for a given hour/year</li>
                <li>Go to <strong>Results</strong> — view charts, evaluation
                    metrics, and download the forecast results</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h3 style='margin-top:0;'>Project Details</h3>
        """, unsafe_allow_html=True)
        details = {
            "👩‍🎓 Researcher"  : "Jackline Boniphace",
            "🏛 Institution"  : "DIT, Dar es Salaam",
            "📘 Programme"   : "M.Eng. Sustainable Energy",
            "📍 Study Area"  : "Ilala District (4 substations)",
            "🎯 Target"      : "Total District Load (MW)",
            "📅 Data Period" : "2019 – 2025 (8-yr series)",
            "⏱ Resolution"  : "Hourly, with 1-year memory",
            "🎯 Horizon"     : "Long-term (growth-aware)",
            "📡 Data Source" : "TANESCO / TMA / NBS",
        }
        for k, v in details.items():
            st.markdown(f"""
            <div style='display:flex; justify-content:space-between;
                        padding:0.4rem 0; border-bottom:1px solid #1F4E8C;
                        font-size:0.88rem;'>
                <span style='color:#9DB8DD;'>{k}</span>
                <span style='color:#FFFFFF; font-weight:600;'>{v}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
#  PAGE — REGISTRATION & UPLOAD
# ════════════════════════════════════════════════════════════════════
def page_registration():
    st.markdown('<div class="section-header">📋 User Registration & Data Upload</div>',
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-card"><h3 style='margin-top:0;'>
        👤 User Registration</h3>""", unsafe_allow_html=True)
        with st.form("registration_form"):
            full_name    = st.text_input("Full Name *")
            email        = st.text_input("Email Address *")
            organisation = st.text_input("Organisation / Department")
            role    = st.selectbox("Role", ["Select role", "Power System Engineer",
                        "Researcher / Student", "Energy Planner",
                        "Grid Operator", "Other"])
            purpose = st.selectbox("Purpose of Use", ["Select purpose",
                        "Load Planning & Scheduling", "Academic Research",
                        "Grid Management", "Policy Planning", "Other"])
            district = st.selectbox("Target District",
                        ["Ilala", "Kinondoni", "Temeke", "Ubungo", "Kigamboni"])
            agree = st.checkbox("I agree that predictions are for planning purposes only")
            submitted = st.form_submit_button("Register & Proceed")
            if submitted:
                if full_name and email and role != "Select role" and agree:
                    st.session_state.user_data = {
                        'full_name': full_name, 'email': email,
                        'organisation': organisation, 'role': role,
                        'purpose': purpose, 'district': district,
                        'registered_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    }
                    st.success(f"Registration successful! Welcome, {full_name}.")
                elif not agree:
                    st.warning("Please accept the terms to proceed.")
                else:
                    st.error("Please fill in all required fields (*).")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card"><h3 style='margin-top:0;'>
        📂 Upload Data for Forecasting</h3>
        <p style='font-size:0.88rem;'>
        Supported formats: CSV, Excel (.xlsx), JSON.</p>""",
        unsafe_allow_html=True)

        uploaded = st.file_uploader("Choose a file",
                    type=['csv', 'xlsx', 'xls', 'json'])
        if uploaded is not None:
            try:
                if uploaded.name.endswith('.csv'):
                    df = pd.read_csv(uploaded)
                elif uploaded.name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(uploaded)
                else:
                    df = pd.read_json(uploaded)
                st.session_state.uploaded_df = df
                st.success(f"File uploaded: **{uploaded.name}** — "
                           f"{len(df):,} rows × {len(df.columns)} columns")
                st.dataframe(df.head(5), use_container_width=True)

                assets    = load_model_assets()
                feat_cols = assets.get('feature_cols', [])
                if feat_cols:
                    missing = [c for c in feat_cols if c not in df.columns]
                    if missing:
                        st.warning(f"Missing columns: {', '.join(missing[:5])}"
                                   + (" ..." if len(missing) > 5 else ""))
                    else:
                        st.markdown("""<div class='success-box'>
                        ✓ All required columns present. Proceed to Prediction.
                        </div>""", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error reading file: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
#  PAGE — PREDICTION
# ════════════════════════════════════════════════════════════════════
def page_prediction():
    st.markdown('<div class="section-header">🤖 Model Prediction & Forecasting</div>',
                unsafe_allow_html=True)

    assets     = load_model_assets()
    model      = assets.get('model')
    scaler_X   = assets.get('scaler_X')
    scaler_y   = assets.get('scaler_y')
    feat_cols  = assets.get('feature_cols', [])
    metadata   = assets.get('metadata', {})
    model_name = metadata.get('best_model', 'ML Model') if metadata else 'ML Model'

    if model is None:
        st.error("Model not found. Ensure 'best_model_deployment/' is next to app.py "
                  "and contains best_model.pkl (or .keras), scaler_X.pkl, scaler_y.pkl, "
                  "feature_columns.pkl and metadata.json.")
        return

    tab1, tab2 = st.tabs(["📊 Upload-based Prediction", "✍ Manual Input Prediction"])

    # ── TAB 1: Upload-based ──────────────────────────────────────
    with tab1:
        if st.session_state.uploaded_df is None:
            st.info("Please upload a data file in the Registration & Upload page first.")
        else:
            df = st.session_state.uploaded_df.copy()
            if 'Datetime' in df.columns:
                df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
            missing_cols = [c for c in feat_cols if c not in df.columns]

            c1, c2, c3 = st.columns(3)
            for col, val, lbl in [
                (c1, f"{len(df):,}", "Rows Uploaded"),
                (c2, model_name.split()[0], "Active Model"),
                (c3, "Ready" if not missing_cols else "Missing Columns", "Data Status"),
            ]:
                with col:
                    st.markdown(f"""<div class="metric-card">
                    <div class="metric-value" style='font-size:1.4rem;'>{val}</div>
                    <div class="metric-label">{lbl}</div></div>""",
                    unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            if missing_cols:
                st.error(f"Cannot predict — missing columns: {missing_cols}")
            else:
                mode = st.radio(
                    "Choose what to do with the uploaded data:",
                    ["📌 Predict on Uploaded Data",
                     "🔮 Forecast Future Years (2030, 2050, ...)"],
                    horizontal=True, key="upload_mode",
                )

                # ---- Mode A: predict on the rows already in the file ----
                if mode == "📌 Predict on Uploaded Data":
                    X = df[feat_cols].ffill().fillna(0).values
                    if st.button("▶  Run Prediction", key="run_pred"):
                        with st.spinner("Running forecast..."):
                            try:
                                preds  = make_prediction(model, X, metadata,
                                                         scaler_y, scaler_X)
                                offset = len(df) - len(preds)
                                result_df = df.iloc[offset:].reset_index(drop=True)
                                result_df['Predicted_Total_District_MW'] = preds.round(3)
                                st.session_state.predictions = result_df
                                st.success(f"Forecast complete — "
                                           f"{len(preds):,} predictions generated.")
                            except Exception as e:
                                st.error(f"Prediction error: {e}")

                    if st.session_state.predictions is not None:
                        res = st.session_state.predictions
                        st.markdown("### 📋 Prediction Results")
                        display_cols = [c for c in
                            ['Datetime','Hour_of_Day','Day_of_Week','Month',
                             'Temperature_C','Humidity_pct','Total_District_MW',
                             'Predicted_Total_District_MW'] if c in res.columns]
                        st.dataframe(res[display_cols].head(50),
                                     use_container_width=True)
                        d1, d2 = st.columns(2)
                        with d1:
                            st.download_button("Download CSV",
                                data=res.to_csv(index=False).encode(),
                                file_name="ilala_forecasting_results.csv",
                                mime="text/csv")
                        with d2:
                            st.download_button("Download Excel",
                                data=to_excel_bytes(res),
                                file_name="ilala_forecasting_results.xlsx",
                                mime="application/vnd.openxmlformats-officedocument"
                                     ".spreadsheetml.sheet")

                # ---- Mode B: extend the upload into a future-year forecast ----
                else:
                    required_for_forecast = ['Datetime', 'Total_District_MW']
                    missing_req = [c for c in required_for_forecast
                                   if c not in df.columns or df[c].isna().all()]
                    if missing_req:
                        st.error("Future-year forecasting needs a valid "
                                 f"{', '.join(required_for_forecast)} column "
                                 f"in your upload. Missing/empty: {missing_req}")
                    else:
                        last_dt = df['Datetime'].max()
                        st.caption(f"Your uploaded data ends at **{last_dt}**. "
                                   "Pick how far into the future to forecast — "
                                   "the model will roll forward hour by hour, "
                                   "feeding each prediction back in as the "
                                   "history for the next hour.")

                        preset = st.radio("Quick presets", ["2030", "2040", "2050", "Custom"],
                                           horizontal=True, key="year_preset")
                        if preset == "Custom":
                            target_year = st.number_input(
                                "Target year", min_value=last_dt.year + 1,
                                max_value=2100, value=min(2035, 2100), step=1)
                        else:
                            target_year = int(preset)

                        n_est_hours = max(0, int((pd.Timestamp(target_year, 12, 31, 23)
                                                   - last_dt).total_seconds() // 3600))
                        st.caption(f"≈ {n_est_hours:,} hourly predictions will be "
                                   f"generated, from {last_dt + pd.Timedelta(hours=1)} "
                                   f"through {target_year}-12-31 23:00. "
                                   "Longer horizons take proportionally longer to run "
                                   "(roughly 1–2 minutes per 10 years).")
                        st.caption("⚠ Weather is projected from historical seasonal "
                                   "averages, socio-economic drivers (population, "
                                   "customers, connections...) follow their historical "
                                   "growth trend, and holidays beyond your uploaded "
                                   "data use a fixed annual calendar — treat results "
                                   "as a growth-aware scenario, not a guarantee.")

                        if target_year <= last_dt.year:
                            st.warning("Target year must be after the last year "
                                       "in your uploaded data.")
                        elif st.button(f"🚀  Run Long-Term Forecast to {target_year}",
                                       key="run_future_forecast"):
                            progress_bar = st.progress(0.0)
                            status_text = st.empty()
                            def _cb(frac):
                                progress_bar.progress(min(1.0, frac))
                                status_text.text(f"Forecasting... {frac*100:,.0f}%")
                            try:
                                with st.spinner("Running long-term forecast — "
                                                 "this may take a little while for "
                                                 "distant target years..."):
                                    future_df = run_long_term_forecast(
                                        df, target_year, feat_cols, model,
                                        metadata, scaler_X, scaler_y,
                                        progress_cb=_cb)
                                progress_bar.empty()
                                status_text.empty()
                                st.session_state.future_predictions = future_df
                                st.success(f"Long-term forecast complete — "
                                           f"{len(future_df):,} hourly predictions "
                                           f"generated through {target_year}.")
                            except Exception as e:
                                progress_bar.empty()
                                status_text.empty()
                                st.error(f"Forecast error: {e}")

                        if st.session_state.future_predictions is not None:
                            fut = st.session_state.future_predictions
                            st.markdown("### 📈 Future Forecast Results")

                            fig = go.Figure()
                            hist_tail = df.sort_values('Datetime').tail(24*30)
                            fig.add_trace(go.Scatter(
                                x=hist_tail['Datetime'], y=hist_tail['Total_District_MW'],
                                mode='lines', name='Historical (last 30 days)',
                                line=dict(color='#4DA3FF')))
                            fig.add_trace(go.Scatter(
                                x=fut['Datetime'], y=fut['Predicted_Total_District_MW'],
                                mode='lines', name=f'Forecast to {target_year}',
                                line=dict(color='#F5B041')))
                            fig.update_layout(**DARK_CHART, height=420,
                                title="Historical Tail vs. Long-Term Forecast",
                                legend=dict(orientation='h', y=1.1))
                            st.plotly_chart(dark_axes(fig), use_container_width=True)

                            st.dataframe(
                                fut[['Datetime', 'Predicted_Total_District_MW']].head(50),
                                use_container_width=True)

                            d1, d2 = st.columns(2)
                            with d1:
                                st.download_button("Download CSV",
                                    data=fut.to_csv(index=False).encode(),
                                    file_name=f"ilala_forecast_to_{target_year}.csv",
                                    mime="text/csv")
                            with d2:
                                st.download_button("Download Excel",
                                    data=to_excel_bytes(fut),
                                    file_name=f"ilala_forecast_to_{target_year}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument"
                                         ".spreadsheetml.sheet")

    # ── TAB 2: Manual input — arranged in clear sections ─────────
    with tab2:
        st.markdown("#### Enter values for each factor, then press Predict")
        st.caption("Predicts **Total District Load** (Ilala + Gongo la Mboto + "
                   "Kipawa + NCC substations combined).")

        # SECTION 1 — Time information
        st.markdown('<div class="group-title">🕐 SECTION 1 — Time Information</div>',
                    unsafe_allow_html=True)
        t1, t2, t3, t4 = st.columns(4)
        with t1:
            hour  = st.slider("Hour of Day", 0, 23, 12)
            month = st.slider("Month", 1, 12, 6)
        with t2:
            dow = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 1)
            doy = st.slider("Day of Year", 1, 365, 150)
        with t3:
            season  = st.selectbox("Season", [1, 2, 3, 4],
                        format_func=lambda x: {
                            1: 'Hot & Wet (Dec-Feb)',
                            2: 'Long Rains (Mar-May)',
                            3: 'Cool & Dry (Jun-Aug)',
                            4: 'Short Rains (Sep-Nov)'}[x])
            weekend = st.selectbox("Is Weekend", [0, 1],
                        format_func=lambda x: "Yes" if x else "No")
        with t4:
            holiday = st.selectbox("Is Holiday", [0, 1],
                        format_func=lambda x: "Yes" if x else "No")
            year = st.number_input("Year (for long-term trend)", 2019, 2100, 2025, 1)
            st.caption("Sin/cos encodings, Is_Workday and Temp_x_Hour "
                       "are computed automatically.")
            if year > 2025:
                st.warning(
                    f"⚠ {year} is beyond the training data (2018–2025). "
                    "XGBoost is tree-based and doesn't extrapolate trends the way "
                    "a linear model would — for a meaningful long-term projection, "
                    "also increase Population, Households, Total Customers and "
                    "Connections below to values consistent with that future year.",
                    icon="⚠️"
                )

        # SECTION 2 — Weather
        st.markdown('<div class="group-title">🌡 SECTION 2 — Weather Conditions</div>',
                    unsafe_allow_html=True)
        w1, w2, w3 = st.columns(3)
        with w1:
            temp = st.slider("Temperature (°C)", 20.0, 35.0, 27.0, 0.1)
        with w2:
            humidity = st.slider("Humidity (%)", 60.0, 95.0, 77.0, 0.5)
        with w3:
            rainfall = st.slider("Rainfall (mm)", 0.0, 600.0, 50.0, 1.0)

        # SECTION 3 — Historical load (short-term lags), now on Total District Load
        st.markdown('<div class="group-title">⏱ SECTION 3 — Historical Total '
                    'District Load</div>', unsafe_allow_html=True)
        st.caption("Enter recent TOTAL district load (sum of all 4 substations), "
                   "not a single substation reading.")
        l1, l2, l3, l4, l5, l6 = st.columns(6)
        with l1: lag1h   = st.number_input("Lag 1h (MW)",   1.0, 250.0, 130.0, 1.0)
        with l2: lag2h   = st.number_input("Lag 2h (MW)",   1.0, 250.0, 130.0, 1.0)
        with l3: lag3h   = st.number_input("Lag 3h (MW)",   1.0, 250.0, 130.0, 1.0)
        with l4: lag24h  = st.number_input("Lag 24h (MW)",  1.0, 250.0, 130.0, 1.0)
        with l5: lag48h  = st.number_input("Lag 48h (MW)",  1.0, 250.0, 130.0, 1.0)
        with l6: lag168h = st.number_input("Lag 168h (MW)", 1.0, 250.0, 128.0, 1.0)

        # SECTION 4 — Rolling statistics (short-term)
        st.markdown('<div class="group-title">📊 SECTION 4 — Rolling '
                    'Statistics</div>', unsafe_allow_html=True)
        r1, r2, r3, r4 = st.columns(4)
        with r1: roll3   = st.number_input("Rolling 3h Mean (MW)",  1.0, 250.0, 130.0, 1.0)
        with r2: roll6   = st.number_input("Rolling 6h Mean (MW)",  1.0, 250.0, 130.0, 1.0)
        with r3: roll24  = st.number_input("Rolling 24h Mean (MW)", 1.0, 250.0, 130.0, 1.0)
        with r4: roll24s = st.number_input("Rolling 24h Std (MW)",  0.0, 40.0,  10.0, 0.5)

        # SECTION 5 — NEW: Long-term historical-load memory
        st.markdown('<div class="group-title">📅 SECTION 5 — Long-Term Load '
                    'Memory</div>', unsafe_allow_html=True)
        st.caption("Total district load one full year ago, and the trailing "
                   "1-year rolling average — captures year-over-year growth.")
        y1, y2 = st.columns(2)
        with y1: lag8760h = st.number_input("Load 1 Year Ago (MW)", 1.0, 250.0, 120.0, 1.0)
        with y2: roll8760h = st.number_input("1-Year Rolling Mean Load (MW)", 1.0, 250.0, 125.0, 1.0)

        # SECTION 6 — NEW: Long-term socio-economic drivers
        st.markdown('<div class="group-title">🏙 SECTION 6 — Long-Term '
                    'Socio-Economic Drivers</div>', unsafe_allow_html=True)
        st.caption("Population, connections and customer growth — the key "
                   "long-term factors identified in Objective 1.")
        st.caption("Ranges are widened beyond the historical dataset so you can "
                   "model longer-term growth scenarios (e.g. 2030, 2050).")
        se1, se2, se3 = st.columns(3)
        with se1:
            population = st.number_input("Population", 1_600_000, 10_000_000, 1_800_000, 1000)
            households = st.number_input("Households", 400_000, 3_000_000, 500_000, 500)
        with se2:
            total_customers = st.number_input("Total Customers", 300_000, 2_000_000, 355_000, 100)
            service_conn    = st.number_input("New Service Connections (this year)", 1_000, 300_000, 30_000, 100)
        with se3:
            cumulative_conn = st.number_input("Cumulative Connections (since 2018)", 5_000, 5_000_000, 100_000, 500)
            t1_luku = st.number_input("T1 (Residential LUKU) Customers", 300_000, 2_000_000, 350_000, 100)
        se4, se5 = st.columns(2)
        with se4:
            t2 = st.number_input("T2 (Commercial) Customers", 500, 5_000, 615, 1)
        with se5:
            t3 = st.number_input("T3 (Industrial) Customers", 100, 2_000, 165, 1)

        # PREDICT button — centred
        st.markdown("<br>", unsafe_allow_html=True)
        b1, b2, b3 = st.columns([1, 1.2, 1])
        with b2:
            predict_clicked = st.button("⚡  PREDICT TOTAL DISTRICT LOAD", key="manual_pred")

        if predict_clicked:
            hour_sin = np.sin(2*np.pi*hour/24)
            year_index = year - BASE_YEAR
            input_row = {
                'Hour_of_Day': hour, 'Day_of_Week': dow, 'Month': month,
                'Season': season, 'Is_Weekend': weekend, 'Is_Holiday': holiday,
                'Is_Workday': int(not weekend and not holiday),
                'Hour_sin': hour_sin, 'Hour_cos': np.cos(2*np.pi*hour/24),
                'Month_sin': np.sin(2*np.pi*month/12),
                'Month_cos': np.cos(2*np.pi*month/12),
                'DOW_sin': np.sin(2*np.pi*dow/7),
                'DOW_cos': np.cos(2*np.pi*dow/7),
                'DOY_sin': np.sin(2*np.pi*doy/365),
                'DOY_cos': np.cos(2*np.pi*doy/365),
                'Year_index': year_index,
                'Temperature_C': temp, 'Humidity_pct': humidity,
                'Rainfall_mm': rainfall, 'Temp_x_Hour': temp * hour_sin,
                'Total_District_MW_lag1h': lag1h, 'Total_District_MW_lag2h': lag2h,
                'Total_District_MW_lag3h': lag3h, 'Total_District_MW_lag24h': lag24h,
                'Total_District_MW_lag48h': lag48h, 'Total_District_MW_lag168h': lag168h,
                'Total_District_MW_roll3h_mean': roll3,
                'Total_District_MW_roll6h_mean': roll6,
                'Total_District_MW_roll24h_mean': roll24,
                'Total_District_MW_roll24h_std': roll24s,
                'Total_District_MW_lag8760h': lag8760h,
                'Total_District_MW_roll8760h_mean': roll8760h,
                'Population': population, 'Households': households,
                'Total_Customers': total_customers,
                'Service_Line_Connections': service_conn,
                'Cumulative_Connections': cumulative_conn,
                'T1_LUKU': t1_luku, 'T2': t2, 'T3': t3,
            }
            row_df  = pd.DataFrame([input_row])
            missing = [c for c in feat_cols if c not in row_df.columns]
            if missing:
                st.error(f"Internal error — missing factors: {missing}")
            else:
                X_input = row_df[feat_cols].values
                try:
                    if 'LSTM' in model_name:
                        seq  = np.tile(X_input, (24, 1))
                        X_sc = scaler_X.transform(seq)
                        y_sc = model.predict(X_sc[np.newaxis, :, :],
                                             verbose=0).ravel()
                        pred = float(scaler_y.inverse_transform(
                                     y_sc.reshape(-1, 1)).ravel()[0])
                        st.info("Note: LSTM uses a 24-hour sequence — "
                                "single-hour input is used as a proxy.")
                    else:
                        pred = float(model.predict(X_input)[0])

                    st.markdown(f"""
                    <div style='text-align:center; background:#122E5C;
                                padding:2rem; border-radius:16px;
                                box-shadow:0 6px 20px rgba(0,0,0,0.4);
                                border:1px solid #4DA3FF; margin-top:1rem;'>
                        <div style='font-size:0.9rem; color:#9DB8DD;'>
                            Predicted Total Electricity Load — Ilala District
                            (4 substations combined)</div>
                        <div style='font-size:3.5rem; font-weight:800;
                                    color:#FFFFFF; margin:0.5rem 0;'>
                            {pred:.2f} <span style='font-size:1.5rem;'>MW</span>
                        </div>
                        <div style='font-size:0.85rem; color:#4DA3FF;'>
                            Model: {model_name}  |  Year: {year}  |
                            Hour: {hour:02d}:00  |  Temp: {temp}°C</div>
                    </div>""", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

# ════════════════════════════════════════════════════════════════════
#  PAGE — RESULTS
# ════════════════════════════════════════════════════════════════════
def page_results():
    st.markdown('<div class="section-header">📈 Results & Implementation Insights</div>',
                unsafe_allow_html=True)
    if st.session_state.predictions is None:
        st.info("No predictions yet. Please run a forecast in the Prediction page first.")
        return

    res        = st.session_state.predictions
    assets     = load_model_assets()
    metadata   = assets.get('metadata', {})
    model_name = metadata.get('best_model', 'ML Model') if metadata else 'ML Model'
    has_actual = 'Total_District_MW' in res.columns
    y_pred     = res['Predicted_Total_District_MW'].values

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, lbl in [
        (c1, f"{y_pred.mean():.2f}", "Avg Forecast (MW)"),
        (c2, f"{y_pred.max():.2f}",  "Peak Forecast (MW)"),
        (c3, f"{y_pred.min():.2f}",  "Min Forecast (MW)"),
        (c4, f"{len(y_pred):,}",     "Hours Forecasted"),
        (c5, model_name.split()[0],  "Model Used"),
    ]:
        with col:
            st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style='font-size:1.5rem;'>{val}</div>
            <div class="metric-label">{lbl}</div></div>""",
            unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Time series
    st.markdown("### 📊 Forecast Time Series")
    fig_ts = go.Figure()
    if has_actual:
        fig_ts.add_trace(go.Scatter(y=res['Total_District_MW'].values, name='Actual Load',
                         line=dict(color='#E8EEF7', width=2)))
    fig_ts.add_trace(go.Scatter(y=y_pred, name='Predicted Load',
                     line=dict(color='#4DA3FF', width=2, dash='dash')))
    fig_ts.update_layout(
        title=f'Total District Load Forecast — Ilala District ({model_name})',
        xaxis_title='Time Step (Hours)', yaxis_title='Load (MW)',
        hovermode='x unified', height=420,
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1),
        **DARK_CHART)
    dark_axes(fig_ts)
    st.plotly_chart(fig_ts, use_container_width=True)

    # Scatter
    if has_actual:
        st.markdown("### 🎯 Scatter Plot — Predicted vs Actual Load")
        y_actual = res['Total_District_MW'].values
        metrics  = compute_metrics(y_actual, y_pred)
        lim_min  = min(y_actual.min(), y_pred.min()) - 1
        lim_max  = max(y_actual.max(), y_pred.max()) + 1

        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(x=y_actual, y=y_pred, mode='markers',
            name='Predictions',
            marker=dict(color='#4DA3FF', size=5, opacity=0.55)))
        fig_sc.add_trace(go.Scatter(x=[lim_min, lim_max], y=[lim_min, lim_max],
            mode='lines', name='Perfect Fit (y = x)',
            line=dict(color='#FF6B6B', width=2, dash='dash')))
        z = np.polyfit(y_actual, y_pred, 1)
        x_fit = np.linspace(lim_min, lim_max, 100)
        fig_sc.add_trace(go.Scatter(x=x_fit, y=np.poly1d(z)(x_fit),
            mode='lines', name='Trend Line',
            line=dict(color='#FFD166', width=2)))
        fig_sc.update_layout(
            title=(f'Predicted vs Actual Load — {model_name}<br>'
                   f'<sup>MAE={metrics["MAE"]:.3f} MW | '
                   f'RMSE={metrics["RMSE"]:.3f} MW | '
                   f'MAPE={metrics["MAPE"]:.2f}% | '
                   f'R²={metrics["R2"]:.4f}</sup>'),
            xaxis_title='Actual Load (MW)', yaxis_title='Predicted Load (MW)',
            height=480, **DARK_CHART)
        dark_axes(fig_sc)
        st.plotly_chart(fig_sc, use_container_width=True)

    # Hourly average
    if 'Hour_of_Day' in res.columns:
        st.markdown("### 🕐 Average Forecast by Hour of Day")
        hourly_pred = res.groupby('Hour_of_Day')['Predicted_Total_District_MW'].mean()
        fig_hr = go.Figure()
        fig_hr.add_trace(go.Bar(x=hourly_pred.index, y=hourly_pred.values,
            name='Avg Predicted Load', marker_color='#4DA3FF', opacity=0.85))
        if has_actual:
            hourly_act = res.groupby('Hour_of_Day')['Total_District_MW'].mean()
            fig_hr.add_trace(go.Scatter(x=hourly_act.index, y=hourly_act.values,
                name='Avg Actual Load', mode='lines+markers',
                line=dict(color='#FF6B6B', width=2)))
        fig_hr.update_layout(title='Average Predicted Load by Hour of Day',
            xaxis_title='Hour of Day', yaxis_title='Average Load (MW)',
            height=360, **DARK_CHART)
        dark_axes(fig_hr)
        fig_hr.update_xaxes(tickmode='linear', dtick=1)
        st.plotly_chart(fig_hr, use_container_width=True)

    # Annual / long-term growth view
    if 'Year' in res.columns:
        st.markdown("### 📈 Annual Average — Long-Term Growth View")
        annual_pred = res.groupby('Year')['Predicted_Total_District_MW'].mean()
        fig_yr = go.Figure()
        fig_yr.add_trace(go.Bar(x=annual_pred.index.astype(str), y=annual_pred.values,
            name='Avg Predicted Load', marker_color='#4DA3FF'))
        if has_actual:
            annual_act = res.groupby('Year')['Total_District_MW'].mean()
            fig_yr.add_trace(go.Scatter(x=annual_act.index.astype(str), y=annual_act.values,
                name='Avg Actual Load', mode='lines+markers',
                line=dict(color='#FF6B6B', width=2)))
        fig_yr.update_layout(title='Annual Average Total District Load',
            xaxis_title='Year', yaxis_title='Average Load (MW)',
            height=360, **DARK_CHART)
        dark_axes(fig_yr)
        st.plotly_chart(fig_yr, use_container_width=True)

    # Downloads
    st.markdown("### ⬇ Download Full Results")
    d1, d2 = st.columns(2)
    with d1:
        st.download_button("📄 Download Results CSV",
            data=res.to_csv(index=False).encode(),
            file_name="ilala_forecast_final.csv", mime="text/csv")
    with d2:
        st.download_button("📊 Download Results Excel",
            data=to_excel_bytes(res),
            file_name="ilala_forecast_final.xlsx",
            mime="application/vnd.openxmlformats-officedocument"
                 ".spreadsheetml.sheet")

# ════════════════════════════════════════════════════════════════════
#  SIDEBAR  (shown after authentication only)
# ════════════════════════════════════════════════════════════════════
def sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding:1rem 0 1.5rem;'>
            <div style='font-size:2.5rem;'>⚡</div>
            <div style='font-size:1rem; font-weight:700; color:white;'>
                Ilala Load Forecasting</div>
            <div style='font-size:0.75rem; color:#9DB8DD; margin-top:0.3rem;'>
                DIT — M.Eng. 2026</div>
        </div>""", unsafe_allow_html=True)

        if st.session_state.get('username'):
            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.12); padding:0.6rem;
                        border-radius:8px; text-align:center;
                        margin-bottom:1rem; font-size:0.85rem;'>
                👤 {st.session_state.get('username')}</div>""",
                unsafe_allow_html=True)

        pages = {
            "ℹ️ About System" : "About",
            "📋 Registration" : "Registration",
            "🤖 Prediction"   : "Prediction",
            "📈 Results"      : "Results",
        }
        st.markdown("<br>", unsafe_allow_html=True)
        for label, page in pages.items():
            if st.button(label, key=f"nav_{page}"):
                st.session_state.current_page = page
                st.rerun()

        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("🚪 Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        st.markdown("""
        <div style='position:absolute; bottom:1rem; left:0; right:0;
                    text-align:center; font-size:0.72rem; color:#9DB8DD;'>
            Jackline Boniphace<br>M.Eng. Sustainable Energy<br>
            DIT, Dar es Salaam 2026</div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
#  MAIN ROUTER
# ════════════════════════════════════════════════════════════════════
def main():
    if not st.session_state.authenticated:
        page_login()
        return

    sidebar()
    page = st.session_state.current_page
    if   page == 'About'        : page_about()
    elif page == 'Registration' : page_registration()
    elif page == 'Prediction'   : page_prediction()
    elif page == 'Results'      : page_results()
    else                        : page_about()

if __name__ == '__main__':
    main()