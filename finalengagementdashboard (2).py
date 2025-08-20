

#Streamlit dashboard for Engagement Trend Analysis (2021–2025)
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Optional, Dict, Iterable, Union, IO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import streamlit as st

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

st.set_page_config(
    page_title="Engagement Trend Dashboard (2021–2025)",
    page_icon="📈",
    layout="wide"
)

PRIMARY    = "#2563EB"  
PRIMARY_LT = "#60A5FA" 
PRIMARY_DK = "#1E40AF"  
GREEN      = "#10B981"  
RED        = "#EF4444"  
AMBER      = "#F59E0B"  
SLATE      = "#111827" 
MUTED      = "#6B7280"  
PAPER      = "#FFFFFF"
GRID       = "#E5E7EB"


st.markdown(
    f"""
    <style>
      :root {{
        --primary: {PRIMARY};
        --primary-lt: {PRIMARY_LT};
        --primary-dk: {PRIMARY_DK};
        --green: {GREEN};
        --red: {RED};
        --amber: {AMBER};
        --text: {SLATE};
        --muted: {MUTED};
        --paper: {PAPER};
        --grid: {GRID};
      }}
      .stApp {{ background: var(--paper); color: var(--text); }}
      .block-container {{ padding-top: .6rem; padding-bottom: .6rem; }}
      [data-testid="stHeader"] {{ background: var(--paper) !important; border-bottom: 1px solid #0000; }}
      [data-testid="stSidebar"] {{ border-right: 1px solid #E5E7EB; }}

      .page-title {{ font-weight: 800; font-size: 1.9rem; letter-spacing: .02em; margin: .2rem 0 1rem 0; }}

      .card {{
        background: #fff; border: 1px solid #E5E7EB; border-radius: 14px;
        box-shadow: 0 10px 24px rgba(0,0,0,.06);
        padding: 14px 16px; margin-bottom: 12px;
      }}
      .kpi-title {{ font-size:.80rem; color: var(--muted); margin-bottom:6px; }}
      .kpi-value {{ font-weight: 800; font-size:1.4rem; color: var(--text); }}
      .kpi-sub {{ font-size:.75rem; color: var(--muted); margin-top:2px; }}

      .viz-card {{ background:#fff; border:1px solid #E5E7EB; border-radius:14px; box-shadow:0 12px 26px rgba(0,0,0,.08); padding: 12px 14px; margin-bottom: 14px; }}
      .viz-title {{ font-weight: 800; font-size: 1.06rem; margin: 2px 0 8px 2px; }}
      .viz-card [data-testid="stPlotlyChart"] > div, .viz-card .js-plotly-plot {{
        filter: drop-shadow(0 10px 18px rgba(0,0,0,.12)); border-radius: 10px; background: #fff;
      }}

      .chip {{ display:inline-block; padding: 2px 8px; border-radius: 999px; font-size:.75rem; font-weight:600; background: #F3F4F6; color: var(--text); margin-left: 6px; }}
      .small {{ font-size:.8rem; color: var(--muted); }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="page-title">📊 Engagement Trend Analysis <span class="chip">2021–2025</span></div>', unsafe_allow_html=True)

DEFAULT_PATHS: Iterable[Path] = (
    Path("./engagement_2021_2025_merged.csv"),
    Path("./data/engagement_2021_2025_merged.csv"),
    Path("/mnt/data/engagement_2021_2025_merged.csv"),
    Path(r"C:\Users\akans\Documents\engagement_2021_2025_merged.csv"),  # original path (optional)
)

def _normalize(col: str) -> str:
    return (
        col.strip()
           .lower()
           .replace(" ", "_")
           .replace("-", "_")
           .replace("/", "_")
           .replace("(", "")
           .replace(")", "")
           .replace(".", "")
           .replace("__", "_")
    )


DATE_CANDIDATES = {"date", "day", "ds", "datetime", "timestamp", "report_date"}
NEW_USERS_CANDS = {"new_users", "newusers", "new_user", "new", "users_new"}
RET_USERS_CANDS = {"returning_users", "returningusers", "returning_user", "users_returning", "ret_users"}
TIME_ENG_CANDS  = {
    "time_engaged_per_session", "time_engaged", "engagement_time",
    "avg_session_time", "average_session_time", "time_per_session",
    "avg_time_per_session", "time_spent_per_session"
}

def _read_csv_any(src: Union[Path, IO[bytes], IO[str]]) -> pd.DataFrame:
    try:
        return pd.read_csv(src)
    except UnicodeDecodeError:
        if hasattr(src, "seek"):  
            src.seek(0)
        return pd.read_csv(src, encoding="utf-8", engine="python", on_bad_lines="skip")

@st.cache_data(show_spinner=False)
def load_data(uploaded_file: Optional[st.runtime.uploaded_file_manager.UploadedFile]) -> pd.DataFrame:
    
    if uploaded_file is not None:
        uploaded_file.seek(0)
        df = _read_csv_any(uploaded_file)
    else:
        found = next((p for p in DEFAULT_PATHS if p.exists()), None)
        if not found:
            st.error("CSV 'engagement_2021_2025_merged.csv' not found. Upload it or place it in ./, ./data/, or /mnt/data/.")
            st.stop()
        df = _read_csv_any(found)

    orig_cols = list(df.columns)
    norm_to_orig = _build_map(orig_cols)
    norm_cols = list(norm_to_orig.keys())

    def pick(candidates: set) -> Optional[str]:
        for n in norm_cols:
            if n in candidates:
                return norm_to_orig[n]
        return None


    date_col = pick(DATE_CANDIDATES)
    new_col  = pick(NEW_USERS_CANDS)
    ret_col  = pick(RET_USERS_CANDS)
    time_col = pick(TIME_ENG_CANDS)

    
    if not date_col and len(orig_cols):
        first = orig_cols[0]
        try:
            test = pd.to_datetime(df[first], errors="coerce")
            if test.notna().mean() > 0.7:
                date_col = first
        except Exception:
            pass

   
    missing = []
    if not date_col: missing.append("date")
    if not new_col:  missing.append("new_users")
    if not ret_col:  missing.append("returning_users")
    if not time_col: missing.append("time_engaged_per_session")

    if missing:
        st.error(
            "Your CSV is missing expected columns (or close variants): "
            + ", ".join(missing)
            + "\n\nHeaders found:\n- " + "\n- ".join(map(str, orig_cols))
        )
        st.stop()

    
    out = df.rename(columns={
        date_col: "date",
        new_col: "new_users",
        ret_col: "returning_users",
        time_col: "time_engaged_per_session",
    }).copy()


    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for c in ["new_users", "returning_users", "time_engaged_per_session"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["month_name"] = out["date"].dt.strftime("%b")
    out["weekday"] = out["date"].dt.day_name()
    out["hour"] = out["date"].dt.hour.fillna(0).astype(int)
    out["total_users"] = out["new_users"].fillna(0) + out["returning_users"].fillna(0)
    out["pct_returning"] = np.where(out["total_users"] > 0, out["returning_users"] / out["total_users"] * 100, np.nan)

    return out


st.sidebar.header("Controls")
uploaded = st.sidebar.file_uploader("Upload engagement_2021_2025_merged.csv", type=["csv"])

with st.spinner("Loading data…"):
    df = load_data(uploaded)

years = sorted(df["year"].unique())
min_y, max_y = min(years), max(years)
yr = st.sidebar.slider("Year range", min_y, max_y, (min_y, max_y))
show_forecast = st.sidebar.toggle("Forecast next 12 months", value=True)
forecast_metric = st.sidebar.selectbox("Forecast metric", ["new_users", "returning_users", "total_users"], index=2)
roll_days = st.sidebar.select_slider("Smoothing (rolling mean)", options=[0, 7, 14, 30], value=7)
show_points = st.sidebar.checkbox("Show points on lines", value=False)

f = df[(df["year"] >= yr[0]) & (df["year"] <= yr[1])].copy()


T1, T2, T3, T4, T5, T6 = st.columns(6)

def kpi(col, title: str, value: str, color: str, sub: str | None = None):
    with col:
        st.markdown(
            f"""
            <div class='card'>
              <div class='kpi-title'>{title}</div>
              <div class='kpi-value' style='color:{color}'>{value}</div>
              {f"<div class='kpi-sub'>{sub}</div>" if sub else ""}
            </div>
            """,
            unsafe_allow_html=True,
        )

if len(f):
    total_users = int(f["total_users"].sum())
    avg_time = float(f["time_engaged_per_session"].mean())
    ret_avg = float(f["pct_returning"].mean())
    best_year = int(f.groupby("year")["time_engaged_per_session"].mean().idxmax())
    mm = f.groupby("month_name")["time_engaged_per_session"].mean().sort_values(ascending=False)
    peak_month, peak_val = (mm.index[0], mm.iloc[0]) if len(mm) else ("–", 0)
    low_month, low_val   = (mm.index[-1], mm.iloc[-1]) if len(mm) else ("–", 0)
else:
    total_users, avg_time, ret_avg, best_year = 0, 0.0, 0.0, "–"
    peak_month, peak_val, low_month, low_val = "–", 0, "–", 0

kpi(T1, "Total users", f"{total_users:,}", PRIMARY)
kpi(T2, "Avg time / session", f"{avg_time:.1f} min", AMBER)
kpi(T3, "% returning (avg)", f"{ret_avg:.1f}%", GREEN)
kpi(T4, "Best year (eng)", f"{best_year}", PRIMARY_DK)
kpi(T5, "Peak month", f"{peak_month} ({peak_val:.1f}m)", PRIMARY_LT)
kpi(T6, "Lowest month", f"{low_month} ({low_val:.1f}m)", MUTED)

st.divider()


TEMPLATE = "plotly_white"

def style_fig(fig: go.Figure, height: int | None = 360):
    fig.update_layout(
        template=TEMPLATE,
        height=height,
        margin=dict(t=40, r=16, b=16, l=16),
        font=dict(color=SLATE, size=13),
        legend=dict(orientation="h", y=1.1, x=1, xanchor="right"),
        plot_bgcolor=PAPER, paper_bgcolor=PAPER,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=SLATE, automargin=True)
    fig.update_yaxes(gridcolor=GRID, linecolor=SLATE, automargin=True)
    return fig

def viz_card(title: str, fig: go.Figure, height: int | None = 360):
    st.markdown(f"<div class='viz-card'><div class='viz-title'>{title}</div>", unsafe_allow_html=True)
    st.plotly_chart(style_fig(fig, height=height), use_container_width=True, theme=None, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)


TAB1, TAB2, TAB3, TAB4 = st.tabs(["Trends", "Growth", "Heatmap", "Forecast"])

# ----- Trends
with TAB1:
    if len(f):
        daily = f.groupby("date")[["new_users", "returning_users", "total_users"]].sum().reset_index()
        if roll_days and roll_days > 0:
            daily[["new_users", "returning_users", "total_users"]] = (
                daily[["new_users", "returning_users", "total_users"]].rolling(roll_days).mean()
            )

       
        fig = go.Figure()
        fig.add_traces([
            go.Scatter(
                x=daily["date"], y=daily["total_users"], name="Total users",
                mode="lines+markers" if show_points else "lines",
                line=dict(width=3, color=PRIMARY), fill="tozeroy", fillcolor="rgba(37,99,235,.12)"
            ),
            go.Scatter(
                x=daily["date"], y=daily["new_users"], name="New users",
                mode="lines+markers" if show_points else "lines",
                line=dict(width=2, color=GREEN)
            ),
            go.Scatter(
                x=daily["date"], y=daily["returning_users"], name="Returning users",
                mode="lines+markers" if show_points else "lines",
                line=dict(width=2, color=AMBER)
            ),
        ])
        fig.update_layout(title="Daily Engagement — Total / New / Returning")
        viz_card("New vs Returning vs Total (Daily)", fig, height=420)

        monthly = f.groupby(["year", "month"])["time_engaged_per_session"].mean().reset_index()
        monthly["month_str"] = monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
        fig2 = px.line(
            monthly, x="month_str", y="time_engaged_per_session",
            markers=show_points, template=TEMPLATE, title="Monthly Avg Time Engaged per Session"
        )
        fig2.update_traces(line=dict(width=3, color=PRIMARY))
        fig2.update_xaxes(tickangle=45)
        viz_card("Monthly Engagement Time", fig2)

        pr = f.groupby(["year", "month"])["pct_returning"].mean().reset_index()
        pr["month_str"] = pr["year"].astype(str) + "-" + pr["month"].astype(str).str.zfill(2)
        fig3 = px.line(
            pr, x="month_str", y="pct_returning",
            markers=show_points, template=TEMPLATE, title="% Returning Users Over Time"
        )
        fig3.update_traces(line=dict(width=3, color=GREEN))
        fig3.update_xaxes(tickangle=45)
        viz_card("% Returning Over Time", fig3)
    else:
        st.info("No data in selected range.")

with TAB2:
    yearly = f.groupby("year")[["new_users", "returning_users"]].sum().reset_index()
    if len(yearly):
        yearly["total_users"] = yearly["new_users"] + yearly["returning_users"]
        fig1 = px.bar(
            yearly, x="year", y=["new_users", "returning_users"], barmode="stack",
            template=TEMPLATE, title="Yearly New vs Returning Users",
            color_discrete_map={"new_users": GREEN, "returning_users": AMBER}
        )
        viz_card("Yearly composition", fig1)

        growth = yearly.copy()
        for col in ["new_users", "returning_users", "total_users"]:
            growth[f"{col}_growth_%"] = growth[col].pct_change() * 100
        fig2 = px.bar(
            growth, x="year",
            y=["new_users_growth_%", "returning_users_growth_%", "total_users_growth_%"],
            barmode="group", template=TEMPLATE, title="Year-over-Year Growth (%)",
            color_discrete_map={
                "new_users_growth_%": GREEN,
                "returning_users_growth_%": AMBER,
                "total_users_growth_%": PRIMARY,
            }
        )
        viz_card("YoY Growth", fig2)
    else:
        st.info("Not enough data for growth charts.")

with TAB3:
    if f["hour"].nunique() > 1:
        heat = f.groupby(["weekday", "hour"])["time_engaged_per_session"].mean().reset_index()
        fig_h = px.density_heatmap(
            heat, x="hour", y="weekday", z="time_engaged_per_session",
            color_continuous_scale="Blues", template=TEMPLATE,
            title="Avg Engagement (min) by Hour & Weekday"
        )
        viz_card("Engagement Heatmap", fig_h, height=440)
    else:
        
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        w = f.groupby("weekday")["time_engaged_per_session"].mean().reindex(order).reset_index()
        fig_w = px.bar(w, x="weekday", y="time_engaged_per_session", template=TEMPLATE,
                       color_discrete_sequence=[PRIMARY])
        fig_w.update_layout(title="Avg Engagement (min) by Weekday")
        viz_card("Engagement by Weekday", fig_w)


with TAB4:
    if show_forecast and len(f):
        ser = f.groupby("date")[forecast_metric].sum().reset_index()
        st.caption("Forecasts are illustrative and assume continuity of prior trends.")
        if PROPHET_AVAILABLE:
            df_fc = ser.rename(columns={"date": "ds", forecast_metric: "y"})
            try:
                m = Prophet()
                m.fit(df_fc)
                future = m.make_future_dataframe(periods=365)
                forecast = m.predict(future)
                fig_fc = px.line(
                    forecast, x="ds", y="yhat", template=TEMPLATE,
                    title=f"Prophet Forecast for {forecast_metric} (next 12 months)"
                )
                fig_fc.update_traces(line=dict(width=3, color=PRIMARY))
                viz_card("Prophet Forecast", fig_fc)
            except Exception:
                st.warning("Prophet failed; falling back to linear regression.")
                
                PROPHET_OK = False
        if not PROPHET_AVAILABLE:
            s = ser.reset_index(drop=True).copy()
            s["t"] = (s["date"] - s["date"].min()).dt.days
            X = s[["t"]].values
            y = s[forecast_metric].fillna(s[forecast_metric].mean()).values
            lr = LinearRegression().fit(X, y)
            future_days = pd.date_range(s["date"].max() + pd.Timedelta(days=1), periods=365, freq="D")
            t_future = (future_days - s["date"].min()).days.values.reshape(-1, 1)
            yhat = lr.predict(t_future)
            out = pd.DataFrame({"date": future_days, "yhat": yhat})
            base = s.rename(columns={forecast_metric: "yhat"})[["date", "yhat"]]
            fig_fc2 = px.line(
                pd.concat([base, out]), x="date", y="yhat", template=TEMPLATE,
                title=f"Linear Regression Forecast for {forecast_metric} (next 12 months)"
            )
            fig_fc2.update_traces(line=dict(width=3, color=PRIMARY))
            viz_card("Linear Regression Forecast", fig_fc2)
    else:
        st.info("Enable 'Forecast next 12 months' to view projections.")


with st.expander("Diagnostics (mapping & sample)"):
    st.markdown(
        "- <span class='small'>If you uploaded a file, it's used first. Otherwise the app searches default paths.</span>",
        unsafe_allow_html=True
    )
    st.write("Columns:", list(df.columns))
    st.dataframe(df.head(10))

st.download_button(
    " Download filtered data (CSV)",
    f.to_csv(index=False).encode("utf-8"),
    "engagement_filtered.csv"
)

with st.expander("About this dashboard"):
    st.markdown(
        """
        **Engagement Trend Analysis (2021–2025)** — refreshed UI with:
        - Clean visual hierarchy and consistent color system
        - Tabs for context-driven exploration (Trends, Growth, Heatmap, Forecast)
        - Optional line-point toggles & rolling-mean smoothing
        - Forecasting via Prophet (if installed) or a linear-regression fallback
        """
    )


