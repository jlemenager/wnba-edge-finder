"""
WNBA Edge Finder  —  Streamlit Community Cloud
================================================
Single-file deployment. All logic is self-contained.

Deploy steps:
  1. Train model: python wnba_model.py
  2. Push app.py + wnba_predictor_v3.joblib + requirements.txt to GitHub
  3. Deploy at share.streamlit.io
  4. Add ODDS_API_KEY to Streamlit Cloud secrets

Local run:
  streamlit run app.py
"""

import os, math, time, warnings
from datetime import datetime, timedelta
from collections import defaultdict

import numpy  as np
import pandas as pd
import joblib
import requests
import streamlit as st
import plotly.graph_objects as go
import scipy.stats as stats

warnings.filterwarnings("ignore")

from nba_api.stats.endpoints import LeagueGameLog
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline  import Pipeline

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
WNBA_LEAGUE_ID = "10"
EMA_SHORT      = 5
EMA_LONG       = 10
ELO_K          = 20
ELO_BASE       = 1500
API_DELAY      = 0.65
MODEL_PATH     = "wnba_predictor_v3.joblib"

TEAM_CITIES = {
    "Atlanta Dream"          : (33.7490,  -84.3880),
    "Chicago Sky"            : (41.8827,  -87.6741),
    "Connecticut Sun"        : (41.4901,  -72.0882),
    "Dallas Wings"           : (32.7903,  -97.0945),
    "Indiana Fever"          : (39.7640,  -86.1555),
    "Las Vegas Aces"         : (36.0905, -115.1853),
    "Los Angeles Sparks"     : (34.0430, -118.2673),
    "Minnesota Lynx"         : (44.9795,  -93.2760),
    "New York Liberty"       : (40.6826,  -74.0017),
    "Phoenix Mercury"        : (33.4457, -112.0712),
    "Portland Fire"          : (45.5051, -122.6750),
    "Seattle Storm"          : (47.6220, -122.3540),
    "Toronto Tempo"          : (43.6435,  -79.3791),
    "Washington Mystics"     : (38.8988,  -77.0207),
    "Golden State Valkyries" : (37.7749, -122.4194),
}

SIGNAL_ORDER = ["HIGH", "STRONG", "VALUE", "MARGINAL", "NOISE"]
SIGNAL_META  = {
    "HIGH"    : {"label": "⚡ HIGH CONVICTION", "color": "#a855f7", "min_edge": 0.09, "min_margin": 7.0},
    "STRONG"  : {"label": "◆ STRONG",           "color": "#0ea5e9", "min_edge": 0.06, "min_margin": 5.0},
    "VALUE"   : {"label": "▲ VALUE",             "color": "#16a34a", "min_edge": 0.04, "min_margin": 3.0},
    "MARGINAL": {"label": "~ MARGINAL",          "color": "#d97706", "min_edge": 0.02, "min_margin": 2.0},
    "NOISE"   : {"label": "— NOISE",             "color": "#475569", "min_edge": 0.0,  "min_margin": 0.0},
}

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="WNBA Edge Finder",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Barlow:wght@300;400;600;700;800&display=swap');

/* ── Global base ── */
html, body, [class*="css"] { font-family: 'Barlow', sans-serif; }

/* ── App title: black ── */
.app-title {
    font-family: 'Barlow', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    letter-spacing: -.01em;
    color: #111827 !important;
    line-height: 1.1;
    margin-bottom: 4px;
}
.app-sub  { font-family: 'Space Mono', monospace; font-size: .72rem; color: #94a3b8; letter-spacing: .06em; }
.run-hint { font-size: .85rem; color: #64748b; margin-top: 6px; }

/* ── Sidebar: original dark theme ── */
section[data-testid="stSidebar"] {
    background: #080d14 !important;
    border-right: 1px solid #1e2d3d;
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; font-weight: 700 !important; }
section[data-testid="stSidebar"] hr { border-color: #1e2d3d !important; }

/* ── Sidebar input fields: black text on light background ── */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] select,
section[data-testid="stSidebar"] textarea {
    color: #111827 !important;
    background: #f9fafb !important;
    border: 1px solid #d1d5db !important;
}
section[data-testid="stSidebar"] .stSelectbox > div > div,
section[data-testid="stSidebar"] .stDateInput > div > div,
section[data-testid="stSidebar"] div[data-baseweb="select"] div,
section[data-testid="stSidebar"] div[data-baseweb="input"] input {
    color: #111827 !important;
    background: #f9fafb !important;
}
/* Dropdown option text */
section[data-testid="stSidebar"] div[role="option"],
section[data-testid="stSidebar"] div[role="listbox"] * {
    color: #111827 !important;
    background: #f9fafb !important;
}

/* ── Game cards (dark style kept for contrast) ── */
.card { background: #0f1722; border: 1px solid #1e2d3d; border-radius: 12px; padding: 20px 22px; margin-bottom: 14px; }
.card.HIGH    { border-left: 4px solid #a855f7; box-shadow: -3px 0 18px rgba(168,85,247,.15); }
.card.STRONG  { border-left: 4px solid #0ea5e9; box-shadow: -3px 0 18px rgba(14,165,233,.12); }
.card.VALUE   { border-left: 4px solid #16a34a; box-shadow: -3px 0 18px rgba(22,163,74,.12); }
.card.MARGINAL{ border-left: 4px solid #d97706; }
.card.NOISE   { border-left: 4px solid #334155; opacity: .75; }
.badge { display: inline-block; padding: 3px 11px; border-radius: 20px; font-family: 'Space Mono', monospace; font-size: 10px; font-weight: 700; letter-spacing: .07em; text-transform: uppercase; margin-bottom: 8px; }
.badge.HIGH    { background: rgba(168,85,247,.15); color: #c084fc; }
.badge.STRONG  { background: rgba(14,165,233,.12); color: #38bdf8; }
.badge.VALUE   { background: rgba(22,163,74,.12);  color: #4ade80; }
.badge.MARGINAL{ background: rgba(217,119,6,.1);   color: #fbbf24; }
.badge.NOISE   { background: #1e2d3d; color: #475569; }
.matchup { font-family: 'Barlow', sans-serif; font-weight: 700; font-size: 1.45rem; color: #f1f5f9; margin: 2px 0 4px; letter-spacing: .01em; }
.matchup .h { color: #60a5fa; }
.matchup .a { color: #f87171; }
.margin-num { font-family: 'Space Mono', monospace; font-size: 2rem; font-weight: 700; }
.margin-num.pos { color: #4ade80; }
.margin-num.neg { color: #f87171; }
.margin-sub { font-size: .72rem; color: #475569; text-transform: uppercase; letter-spacing: .08em; margin-top: -2px; }
.statrow { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
.statbox { background: #0a1018; border: 1px solid #1e2d3d; border-radius: 7px; padding: 8px 12px; min-width: 78px; }
.statbox-val { font-family: 'Space Mono', monospace; font-size: 12px; font-weight: 700; color: #e2e8f0; }
.statbox-val.g { color: #4ade80; }
.statbox-val.r { color: #f87171; }
.statbox-lbl { font-size: 10px; color: #475569; text-transform: uppercase; letter-spacing: .06em; margin-top: 2px; }
.ci-track { background: #1e2d3d; border-radius: 3px; height: 5px; position: relative; margin: 10px 4px 0; }
.ci-fill  { position: absolute; height: 100%; border-radius: 3px; background: rgba(96,165,250,.2); }
.ci-dot   { position: absolute; top: -5px; width: 15px; height: 15px; border-radius: 50%; transform: translateX(-50%); border: 2px solid #080d14; }
.ci-model  { background: #4ade80; }
.ci-spread { background: #f59e0b; }
.ci-lbls { display: flex; justify-content: space-between; margin-top: 5px; font-family: 'Space Mono', monospace; font-size: 9px; color: #334155; }
.oddrow { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; margin-top: 10px; }
.oddlbl { font-size: 10px; color: #334155; text-transform: uppercase; letter-spacing: .07em; }
.chip { font-family: 'Space Mono', monospace; font-size: 10px; padding: 3px 8px; border-radius: 5px; border: 1px solid #1e2d3d; color: #64748b; }
.chip.g { color: #4ade80; border-color: #166534; background: rgba(22,163,74,.08); }
.chip.r { color: #f87171; border-color: #7f1d1d; background: rgba(248,113,113,.06); }
.tier-lbl { font-family: 'Space Mono', monospace; font-size: .68rem; color: #374151; text-transform: uppercase; letter-spacing: .12em; padding: 4px 0 10px; margin-top: 6px; }
.no-games { text-align: center; padding: 70px 20px; color: #374151; font-size: 1rem; font-family: 'Space Mono', monospace; }
.landing p { color: #374151; max-width: 520px; margin: 12px auto 28px; font-size: .95rem; line-height: 1.65; }

/* ── Buttons ── */
div[data-testid="stButton"] > button {
    background: #1a2840; border: 1px solid #2e4a6a; color: #93c5fd;
    border-radius: 8px; padding: 8px 20px;
    font-family: 'Space Mono', monospace; font-size: .8rem;
    letter-spacing: .05em; transition: all .2s;
}
div[data-testid="stButton"] > button:hover {
    background: #1e3048; border-color: #3b82f6; color: #bfdbfe;
}

/* ── Expanders ── */
details summary { color: #111827 !important; font-weight: 600; }
[data-testid="stExpander"] p,
[data-testid="stExpander"] span,
[data-testid="stExpander"] td,
[data-testid="stExpander"] th { color: #111827 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# WNBAPREDICTOR CLASS  — must be defined before joblib.load
# ══════════════════════════════════════════════════════════════════════════════

def _mc(pred, res_params, book_spread=None, n=10000):
    df_t, loc, scale = res_params
    sims = pred + stats.t.rvs(df=df_t, loc=loc, scale=scale, size=n)
    win_prob = float((sims > 0).mean())
    p5, p95  = float(np.percentile(sims, 5)), float(np.percentile(sims, 95))
    result   = dict(predicted_margin=pred, win_probability=win_prob, p5=p5, p95=p95)
    if book_spread is not None:
        result["ats_prob"] = float((sims > book_spread).mean())
        result["edge"]     = result["ats_prob"] - 0.524
    return result

def _ens_pred(ensemble, X, feature_cols):
    Xdf = pd.DataFrame(X.values if isinstance(X, pd.DataFrame) else X, columns=feature_cols)
    w   = ensemble["weights"]
    return (w["xgb"]   * ensemble["xgb_model"].predict(Xdf)
          + w["lgb"]   * ensemble["lgb_model"].predict(Xdf.values)
          + w["ridge"] * ensemble["ridge_pipe"].predict(Xdf.values))

def _dual_pred(dual, X, feature_cols):
    Xdf = pd.DataFrame(X.values if isinstance(X, pd.DataFrame) else X, columns=feature_cols)
    return dual["m_team"].predict(Xdf) - dual["m_opp"].predict(Xdf)

class WNBAPredictor:
    def __init__(self, ensemble, dual, nr_model, calibrator,
                 res_params, feature_cols, alpha=0.7):
        self.ensemble     = ensemble
        self.dual         = dual
        self.nr_model     = nr_model
        self.calibrator   = calibrator
        self.res_params   = res_params
        self.feature_cols = feature_cols
        self.alpha        = alpha

    def predict(self, feature_dict, book_spread=None):
        X    = pd.DataFrame([{c: feature_dict.get(c, 0.0) for c in self.feature_cols}])
        pred = float((self.alpha * _ens_pred(self.ensemble, X, self.feature_cols)
                    + (1 - self.alpha) * _dual_pred(self.dual, X, self.feature_cols))[0])
        mc   = _mc(pred, self.res_params, book_spread)
        cal  = float(self.calibrator.predict([mc["win_probability"]])[0])
        return {
            "predicted_margin"   : round(pred, 2),
            "win_prob_calibrated": round(cal, 4),
            "margin_90ci"        : (round(mc["p5"], 1), round(mc["p95"], 1)),
            "edge_vs_spread"     : round(mc.get("edge", 0.0), 4),
        }

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin(math.radians(lat2-lat1)/2)**2
         + math.cos(p1)*math.cos(p2)*math.sin(math.radians(lon2-lon1)/2)**2)
    return 2*R*math.asin(math.sqrt(a))

def get_coords(name):
    for k, v in TEAM_CITIES.items():
        if any(p.lower() in name.lower() for p in k.split()):
            return v
    return None

def parse_min(v):
    try:
        if isinstance(v, str) and ":" in v:
            a, b = v.split(":"); return int(a) + int(b)/60
        return float(v)
    except: return 0.0

def ema_last(s, span):
    c = s.dropna()
    return float(c.ewm(span=span, adjust=False).mean().iloc[-1]) if not c.empty else 0.0

def american_to_implied(odds):
    odds = int(odds)
    return (100/(odds+100)) if odds > 0 else (abs(odds)/(abs(odds)+100))

# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════════

def fetch_team_logs(season):
    """
    Fetch team game logs for the given season.
    If the current season returns fewer than 10 rows (too early in season,
    API lag, or rate-limited), automatically blends in the prior season so
    EMA features are never built from empty data.
    """
    MIN_ROWS = 10

    def _pull(s):
        time.sleep(API_DELAY)
        try:
            gl = LeagueGameLog(league_id=WNBA_LEAGUE_ID, season=s,
                               season_type_all_star="Regular Season",
                               player_or_team_abbreviation="T")
            df = gl.get_data_frames()[0]
            if df.empty:
                return pd.DataFrame()
            df.columns      = [c.upper() for c in df.columns]
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
            df["IS_HOME"]   = df["MATCHUP"].str.contains(r"\bvs\b", case=False).astype(int)
            df["SEASON"]    = s
            return df
        except:
            return pd.DataFrame()

    current = _pull(season)

    if len(current) >= MIN_ROWS:
        return current

    # Too sparse — blend in prior season as baseline
    prior_season = str(int(season) - 1)
    st.write(f"    ⚠️  {season} has only {len(current)} rows — "
             f"adding {prior_season} as baseline...")
    prior = _pull(prior_season)

    if prior.empty and current.empty:
        return pd.DataFrame()

    combined = pd.concat([prior, current], ignore_index=True)
    combined.sort_values(["TEAM_ID", "GAME_DATE"], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    st.write(f"    ✓  Combined: {len(prior):,} ({prior_season}) + "
             f"{len(current):,} ({season}) rows")
    return combined

def fetch_player_logs(season):
    """
    Fetch player game logs. Falls back to prior season if current is sparse.
    Player logs are used only for lineup strength, so a prior-season
    baseline is much better than empty data.
    """
    MIN_ROWS = 50   # players * games; 50 is a safe minimum

    def _pull(s):
        time.sleep(API_DELAY)
        try:
            gl = LeagueGameLog(league_id=WNBA_LEAGUE_ID, season=s,
                               season_type_all_star="Regular Season",
                               player_or_team_abbreviation="P")
            df = gl.get_data_frames()[0]
            if df.empty:
                return pd.DataFrame()
            df.columns      = [c.upper() for c in df.columns]
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
            df["SEASON"]    = s
            return df
        except:
            return pd.DataFrame()

    current = _pull(season)
    if len(current) >= MIN_ROWS:
        return current

    prior_season = str(int(season) - 1)
    prior = _pull(prior_season)
    if prior.empty and current.empty:
        return pd.DataFrame()
    combined = pd.concat([prior, current], ignore_index=True)
    combined.sort_values(["PLAYER_ID", "GAME_DATE"], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined

def fetch_today_schedule(date_str):
    errors = []

    # Attempt 1: ESPN public API
    try:
        date_compact = date_str.replace("-", "")
        url = (f"https://site.api.espn.com/apis/site/v2/sports/"
               f"basketball/wnba/scoreboard?dates={date_compact}&limit=20")
        r   = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        events = r.json().get("events", [])
        if events:
            games = []
            for event in events:
                comps = event.get("competitions", [{}])[0]
                teams = comps.get("competitors", [])
                if len(teams) < 2: continue
                home = next((t for t in teams if t.get("homeAway") == "home"), teams[0])
                away = next((t for t in teams if t.get("homeAway") == "away"), teams[1])
                status = event.get("status", {}).get("type", {}).get("shortDetail", "Scheduled")
                games.append({
                    "game_id"        : event.get("id", ""),
                    "game_status"    : status,
                    "home_team_id"   : home["team"]["id"],
                    "home_team_name" : home["team"].get("displayName", ""),
                    "home_team_abbr" : home["team"].get("abbreviation", ""),
                    "away_team_id"   : away["team"]["id"],
                    "away_team_name" : away["team"].get("displayName", ""),
                    "away_team_abbr" : away["team"].get("abbreviation", ""),
                })
            if games:
                return games, []
        errors.append(f"ESPN returned 0 events for {date_compact}")
    except Exception as e:
        errors.append(f"ESPN API: {e}")

    # Attempt 2: nba_api LeagueGameLog filtered by date
    try:
        time.sleep(API_DELAY)
        gl = LeagueGameLog(
            league_id=WNBA_LEAGUE_ID,
            season=str(pd.Timestamp(date_str).year),
            season_type_all_star="Regular Season",
            player_or_team_abbreviation="T",
            date_from_nullable=date_str,
            date_to_nullable=date_str,
        )
        df = gl.get_data_frames()[0]
        df.columns = [c.upper() for c in df.columns]
        if not df.empty:
            games = []
            for gid, grp in df.groupby("GAME_ID"):
                if len(grp) < 2: continue
                home_rows = grp[grp["MATCHUP"].str.contains(r"\bvs\b", case=False, na=False)]
                away_rows = grp[~grp["MATCHUP"].str.contains(r"\bvs\b", case=False, na=False)]
                if home_rows.empty or away_rows.empty:
                    home_rows, away_rows = grp.iloc[[0]], grp.iloc[[1]]
                h, a = home_rows.iloc[0], away_rows.iloc[0]
                games.append({
                    "game_id"        : str(gid),
                    "game_status"    : "Scheduled",
                    "home_team_id"   : str(h["TEAM_ID"]),
                    "home_team_name" : str(h["TEAM_NAME"]),
                    "home_team_abbr" : str(h["TEAM_ABBREVIATION"]),
                    "away_team_id"   : str(a["TEAM_ID"]),
                    "away_team_name" : str(a["TEAM_NAME"]),
                    "away_team_abbr" : str(a["TEAM_ABBREVIATION"]),
                })
            if games:
                return games, []
        errors.append("nba_api returned 0 rows for this date")
    except Exception as e:
        errors.append(f"nba_api: {e}")

    return [], errors

def fetch_odds(api_key):
    try:
        r = requests.get(
            "https://api.the-odds-api.com/v4/sports/basketball_wnba/odds/",
            params={"apiKey": api_key, "regions": "us",
                    "markets": "spreads,h2h", "oddsFormat": "american"},
            timeout=10)
        r.raise_for_status()
        return r.json(), r.headers.get("x-requests-remaining", "?")
    except: return [], "?"

def parse_odds_map(raw):
    result = {}
    for g in raw:
        home, away = g.get("home_team", ""), g.get("away_team", "")
        spreads, h2h = {}, {}
        for book in g.get("bookmakers", []):
            for mkt in book.get("markets", []):
                if mkt["key"] == "spreads" and not spreads:
                    for o in mkt.get("outcomes", []):
                        spreads[o["name"]] = {"point": o.get("point", 0), "price": o.get("price", -110)}
                if mkt["key"] == "h2h" and not h2h:
                    for o in mkt.get("outcomes", []):
                        h2h[o["name"]] = o.get("price", -110)
        result[frozenset([home.lower(), away.lower()])] = {
            "home": home, "away": away,
            "home_spread": spreads.get(home, {}).get("point"),
            "away_spread": spreads.get(away, {}).get("point"),
            "home_ml": h2h.get(home), "away_ml": h2h.get(away),
        }
    return result

def match_odds(game, odds_map):
    h, a = game["home_team_name"].lower(), game["away_team_name"].lower()
    key  = frozenset([h, a])
    if key in odds_map: return odds_map[key]
    for k, v in odds_map.items():
        kl = list(k)
        if (any(w in kl[0] or w in kl[1] for w in h.split()) and
            any(w in kl[0] or w in kl[1] for w in a.split())):
            return v
    return None

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def derive_stats(df):
    df = df.copy()
    for c in ["FGM","FG3M","FGA","FTA","TOV","OREB","DREB"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    if "FG3M" not in df.columns: df["FG3M"] = 0.0
    if "FGM"  not in df.columns: df["FGM"]  = 0.0
    if "FGA" in df.columns and df["FGA"].sum() > 0:
        df["EFG_PCT"] = (df["FGM"] + 0.5*df["FG3M"]) / df["FGA"].replace(0, np.nan)
        df["FTR"]     = df["FTA"] / df["FGA"].replace(0, np.nan)
    if all(c in df.columns for c in ["TOV","FGA","FTA"]):
        df["TOV_PCT"] = df["TOV"] / (df["FGA"] + 0.44*df["FTA"] + df["TOV"]).replace(0, np.nan)
    if all(c in df.columns for c in ["FGA","OREB","TOV","FTA"]):
        df["PACE_PROXY"] = df["FGA"] - df["OREB"] + df["TOV"] + 0.44*df["FTA"]
    if "PTS" in df.columns and "PACE_PROXY" in df.columns:
        df["NET_RATING"] = (df["PTS"] / df["PACE_PROXY"].replace(0, np.nan)) * 100
    return df

def compute_elo(team_df):
    base = (team_df[["GAME_ID","TEAM_ID","PTS","GAME_DATE"]]
            .drop_duplicates(subset=["GAME_ID","TEAM_ID"])
            .sort_values("GAME_DATE").reset_index(drop=True))
    pair = (base.merge(base[["GAME_ID","TEAM_ID","PTS"]]
                       .rename(columns={"TEAM_ID":"OPP_TEAM_ID","PTS":"OPP_PTS"}), on="GAME_ID")
            .query("TEAM_ID != OPP_TEAM_ID")
            .sort_values("GAME_DATE").reset_index(drop=True))
    ratings = defaultdict(lambda: float(ELO_BASE))
    done = set()
    for _, row in pair.iterrows():
        tid, oid, gid = str(row["TEAM_ID"]), str(row["OPP_TEAM_ID"]), str(row["GAME_ID"])
        if gid in done: continue
        done.add(gid)
        ra, rb = ratings[tid], ratings[oid]
        ea = 1/(1 + 10**((rb-ra)/400))
        sa = 1.0 if float(row["PTS"]) > float(row["OPP_PTS"]) else 0.0
        ratings[tid] = ra + ELO_K*(sa-ea)
        ratings[oid] = rb + ELO_K*((1-sa)-(1-ea))
    return dict(ratings)

def lineup_strength(team_id, player_df, team_name=""):
    if player_df.empty: return 0.0
    # Try ID match first
    pdf = player_df[player_df["TEAM_ID"].astype(str) == str(team_id)].copy()
    # Fall back to name match if empty (ESPN ID vs nba_api ID mismatch)
    if pdf.empty and team_name and "TEAM_NAME" in player_df.columns:
        for word in team_name.lower().split():
            if len(word) < 3: continue
            mask = player_df["TEAM_NAME"].str.lower().str.contains(word, na=False)
            if mask.any():
                pdf = player_df[mask].copy()
                break
    if pdf.empty: return 0.0
    pdf["MIN_F"] = pdf["MIN"].apply(parse_min) if "MIN" in pdf.columns else 0.0
    for c in ["PTS","REB","AST","STL","BLK","TOV"]:
        if c not in pdf.columns: pdf[c] = 0.0
    pdf["COMP_PM"] = ((pdf["PTS"] + 0.7*pdf["REB"] + pdf["AST"]
                       + pdf["STL"] + pdf["BLK"] - pdf["TOV"])
                      / pdf["MIN_F"].replace(0, np.nan))
    pdf = pdf.sort_values(["PLAYER_ID","GAME_DATE"])
    pdf["EMA_C"] = (pdf.groupby("PLAYER_ID")["COMP_PM"]
                      .transform(lambda x: x.shift(1).ewm(span=EMA_SHORT, adjust=False).mean()))
    pdf["EMA_C"].fillna(pdf["COMP_PM"].median(), inplace=True)
    last = pdf.sort_values("GAME_DATE")["GAME_ID"].iloc[-1] if len(pdf) > 0 else None
    if last is None: return 0.0
    g = pdf[pdf["GAME_ID"] == last]
    if g.empty: return 0.0
    w = g["MIN_F"].clip(lower=0.01)
    return float(np.average(g["EMA_C"].fillna(0), weights=w))

def _find_team_rows(team_logs, team_id, team_name):
    """
    Look up a team's rows from team_logs.
    Tries TEAM_ID first (nba_api IDs). If that returns empty — which happens
    when the schedule came from ESPN and uses different numeric IDs — falls
    back to a case-insensitive partial name match on TEAM_NAME.
    """
    # Try exact ID match first
    rows = team_logs[team_logs["TEAM_ID"].astype(str) == str(team_id)]
    if not rows.empty:
        return rows

    # ID mismatch (e.g. ESPN ID vs nba_api ID) — match by team name
    if "TEAM_NAME" in team_logs.columns and team_name:
        # Try each word in the team name for a partial match
        name_lower = team_name.lower()
        for word in name_lower.split():
            if len(word) < 3:
                continue   # skip short words like "of"
            mask = team_logs["TEAM_NAME"].str.lower().str.contains(word, na=False)
            rows = team_logs[mask]
            if not rows.empty:
                return rows.sort_values("GAME_DATE")

    # Also try TEAM_ABBREVIATION if available
    if "TEAM_ABBREVIATION" in team_logs.columns and team_name:
        abbr = team_name.split()[-1][:3].upper()
        rows = team_logs[team_logs["TEAM_ABBREVIATION"].str.upper() == abbr]
        if not rows.empty:
            return rows.sort_values("GAME_DATE")

    return pd.DataFrame()   # nothing found


def build_fv(team_id, team_name, opp_name, is_home,
             team_logs, player_df, elo_map, opp_elo, feature_cols):
    tl  = _find_team_rows(team_logs, team_id, team_name).sort_values("GAME_DATE")
    tl  = derive_stats(tl)
    e   = lambda col, sp: ema_last(tl[col].dropna(), sp) if col in tl.columns else 0.0
    rest = max(1, min(7, (datetime.today()-tl["GAME_DATE"].iloc[-1]).days)) if len(tl) > 0 else 3
    g7   = int((tl["GAME_DATE"] >= pd.Timestamp(datetime.today()-timedelta(days=7))).sum()) if len(tl) > 0 else 0
    hc   = get_coords(opp_name if not is_home else team_name)
    ac   = get_coords(team_name if not is_home else opp_name)
    trav = haversine_km(*ac, *hc) if (hc and ac and not is_home) else 0.0
    h2h  = float(tl["MARGIN"].mean()) if "MARGIN" in tl.columns and len(tl) > 0 else 0.0
    elo  = elo_map.get(str(team_id), float(ELO_BASE))
    ls   = lineup_strength(team_id, player_df, team_name)
    fv = {
        **{f"EMA{EMA_SHORT}_{c}": e(c, EMA_SHORT) for c in
           ["EFG_PCT","TOV_PCT","OREB_PCT","FTR","PTS","FG_PCT","FG3_PCT","FG3M",
            "OREB","DREB","BLK","FTM","FT_PCT","PACE_PROXY","NET_RATING","OPP_PTS","PACE"]},
        **{f"EMA{EMA_LONG}_{c}":  e(c, EMA_LONG)  for c in
           ["EFG_PCT","TOV_PCT","OREB_PCT","FTR","PTS","FG_PCT","FG3_PCT","FG3M",
            "OREB","DREB","BLK","FTM","FT_PCT","PACE_PROXY","NET_RATING","OPP_PTS","PACE"]},
        "REST_DAYS": rest, "IS_B2B": int(rest==1), "GAMES_LAST_7D": g7,
        "IS_HOME": is_home, "TRAVEL_KM": trav, "H2H_MARGIN": h2h,
        "ELO_DIFF": elo-opp_elo, "ELO_PRE": elo,
        "LINEUP_STRENGTH": ls, "REF_FOUL_TENDENCY": 40.0,
    }
    return {k: fv.get(k, 0.0) for k in feature_cols}

# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def compute_signal(model_margin, book_spread, edge, ci_low, ci_high):
    if book_spread is None or edge is None:
        m = abs(model_margin)
        if m >= 7: return "STRONG"
        if m >= 4: return "VALUE"
        if m >= 2: return "MARGINAL"
        return "NOISE"
    margin_edge  = abs(model_margin - book_spread)
    spread_in_ci = ci_low < book_spread < ci_high
    for lvl in ["HIGH","STRONG","VALUE"]:
        m = SIGNAL_META[lvl]
        if margin_edge >= m["min_margin"] and abs(edge) >= m["min_edge"] and not spread_in_ci:
            return lvl
    if margin_edge >= 2 or abs(edge) >= 0.02: return "MARGINAL"
    return "NOISE"

# ══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(api_key, date_str, predictor):
    feature_cols = predictor.feature_cols

    with st.status("Running analysis...", expanded=True) as status:

        # Step 1: Schedule
        st.write("📅  Step 1 / 5 — Fetching today's schedule...")
        games, sched_errors = fetch_today_schedule(date_str)
        if not games:
            for e in sched_errors:
                st.write(f"    ⚠️  {e}")
            status.update(label="No games found — use manual entry below.", state="error")
            st.session_state["schedule_errors"] = sched_errors
            return []
        st.write(f"    ✓  Found {len(games)} game{'s' if len(games)!=1 else ''}")

        # Step 2: Team stats (with automatic prior-season fallback)
        season = str(pd.Timestamp(date_str).year)
        st.write(f"📊  Step 2 / 5 — Fetching {season} team stats...")
        team_df = fetch_team_logs(season)
        if team_df.empty:
            # Last resort: try one more year back
            fallback = str(int(season) - 2)
            st.write(f"    ⚠️  Both {season} and {str(int(season)-1)} empty — "
                     f"trying {fallback}...")
            team_df = fetch_team_logs(fallback)
        if team_df.empty:
            status.update(label="Could not load team stats from any season.", state="error")
            return []
        seasons_loaded = sorted(team_df["SEASON"].unique()) if "SEASON" in team_df.columns else [season]
        st.write(f"    ✓  {len(team_df):,} rows loaded  (seasons: {seasons_loaded})")

        # Step 3: Player stats (with automatic prior-season fallback)
        st.write("🏃  Step 3 / 5 — Fetching player logs for lineup strength...")
        player_df = fetch_player_logs(season)
        p_seasons = sorted(player_df["SEASON"].unique()) if ("SEASON" in player_df.columns and not player_df.empty) else []
        st.write(f"    ✓  {len(player_df):,} player rows  (seasons: {p_seasons})")

        # Step 4: Odds
        st.write("💰  Step 4 / 5 — Fetching live odds...")
        odds_raw, remaining = fetch_odds(api_key)
        odds_map = parse_odds_map(odds_raw)
        if odds_map:
            st.write(f"    ✓  Odds for {len(odds_map)} game(s)  (requests remaining: {remaining})")
        else:
            st.write("    ⚠️  No odds — predictions will run without spread comparison")

        # Step 5: Predictions
        st.write("🤖  Step 5 / 5 — Running model predictions...")

        elo_map = compute_elo(team_df)
        team_df = derive_stats(team_df)
        opp = (team_df[["GAME_ID","TEAM_ID","PTS","DREB"]]
               .rename(columns={"TEAM_ID":"OPP_TEAM_ID","PTS":"OPP_PTS","DREB":"OPP_DREB"}))
        tdf = team_df.merge(opp, on="GAME_ID", suffixes=("","_y"))
        tdf = tdf[tdf["TEAM_ID"] != tdf["OPP_TEAM_ID"]].copy()
        if "OREB" in tdf.columns and "OPP_DREB" in tdf.columns:
            tdf["OREB_PCT"] = tdf["OREB"] / (tdf["OREB"]+tdf["OPP_DREB"]).replace(0, np.nan)
        if "PTS" in tdf.columns:
            tdf["MARGIN"] = tdf["PTS"] - tdf["OPP_PTS"]

        # Diagnostic: tdf is now built, safe to inspect
        log_ids  = set(tdf["TEAM_ID"].astype(str).unique()) if not tdf.empty else set()
        game_ids = set()
        for g in games:
            game_ids.add(str(g["home_team_id"]))
            game_ids.add(str(g["away_team_id"]))
        matched = log_ids & game_ids
        if not matched:
            st.write(f"    ℹ️  ID mismatch (ESPN vs nba_api) — using name-based matching")
        else:
            st.write(f"    ✓  Team IDs matched directly ({len(matched)} teams)")

        # Build name → nba_api TEAM_ID map so elo lookups work even when
        # the schedule used ESPN IDs (which don't match nba_api IDs)
        name_to_tid = {}
        if "TEAM_NAME" in tdf.columns:
            for _, row in tdf[["TEAM_ID","TEAM_NAME"]].drop_duplicates().iterrows():
                name_to_tid[str(row["TEAM_NAME"]).lower()] = str(row["TEAM_ID"])

        def _elo_for(team_id, team_name):
            # Try direct ID first
            v = elo_map.get(str(team_id))
            if v is not None:
                return v
            # Fall back to name lookup
            tl_rows = _find_team_rows(tdf, team_id, team_name)
            if not tl_rows.empty:
                tid = str(tl_rows.iloc[0]["TEAM_ID"])
                v = elo_map.get(tid)
                if v is not None:
                    return v
            return float(ELO_BASE)

        results = []
        for g in games:
            odds     = match_odds(g, odds_map)
            home_elo = _elo_for(g["home_team_id"], g["home_team_name"])
            away_elo = _elo_for(g["away_team_id"], g["away_team_name"])
            home_fv  = build_fv(g["home_team_id"], g["home_team_name"], g["away_team_name"],
                                1, tdf, player_df, elo_map, away_elo, feature_cols)
            away_fv  = build_fv(g["away_team_id"], g["away_team_name"], g["home_team_name"],
                                0, tdf, player_df, elo_map, home_elo, feature_cols)
            hs  = odds["home_spread"] if odds else None

            # Sign convention fix:
            # Betting spreads: negative = favored (e.g. home -4.5 = home wins by 4.5)
            # Model margin:    positive = home wins (e.g. +4.5 = home wins by 4.5)
            # They're OPPOSITE signs for the same outcome, so we negate hs
            # before passing it to the model so all comparisons are in the
            # same units as model_margin.
            book_line = -hs if hs is not None else None

            hp  = predictor.predict(home_fv, book_spread=book_line)
            ap  = predictor.predict(away_fv, book_spread=(-book_line if book_line is not None else None))

            # hp > 0 means home wins; ap > 0 means AWAY wins (away perspective)
            # To get home margin from both: hp stays, ap gets negated
            model_margin = (hp["predicted_margin"] - ap["predicted_margin"]) / 2
            ci    = hp["margin_90ci"]
            edge  = hp.get("edge_vs_spread")
            signal = compute_signal(model_margin, book_line, edge, ci[0], ci[1])
            home_tdf = tdf[tdf["TEAM_ID"].astype(str) == str(g["home_team_id"])]
            rest_days = max(1, (datetime.today()-home_tdf["GAME_DATE"].max()).days) if len(home_tdf) > 0 else 3
            results.append({
                **g,
                "model_margin"  : round(model_margin, 2),
                "home_win_prob" : hp["win_prob_calibrated"],
                "away_win_prob" : ap["win_prob_calibrated"],
                "margin_90ci"   : ci,
                "edge"          : edge,
                "home_spread"   : hs,           # raw betting spread (-4.5 = home favored)
                "book_line"     : book_line,     # same units as model_margin (+4.5 = home wins 4.5)
                "home_ml"       : odds["home_ml"] if odds else None,
                "away_ml"       : odds["away_ml"] if odds else None,
                "signal"        : signal,
                "has_odds"      : odds is not None,
                "rest_days"     : rest_days,
            })

        results.sort(key=lambda x: SIGNAL_ORDER.index(x["signal"]))
        status.update(label=f"Done — {len(results)} game(s) analysed", state="complete")

    return results

# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def donut(home_prob, h_abbr, a_abbr):
    h = round(home_prob*100, 1); a = round(100-h, 1)
    fig = go.Figure(go.Pie(
        values=[h, a], labels=[h_abbr, a_abbr], hole=0.7,
        direction="clockwise", sort=False,
        marker=dict(colors=["#3b82f6","#ef4444"], line=dict(color="#080d14", width=2)),
        textinfo="none", hoverinfo="label+percent"))
    fig.add_annotation(text=f"<b>{h}%</b>", x=.5, y=.57, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=18, color="#93c5fd", family="Space Mono"))
    fig.add_annotation(text="home", x=.5, y=.4, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=9, color="#475569"))
    fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=0,b=0), height=120)
    return fig

def ci_bar_html(ci_low, ci_high, model_margin, book_spread):
    rng = max(ci_high-ci_low, 1)
    def pct(v): return max(0, min(100, (v-(ci_low-5))/(rng+10)*100))
    bp, bw = pct(ci_low), pct(ci_high)-pct(ci_low)
    mp  = pct(model_margin)
    sp  = pct(book_spread) if book_spread is not None else None
    zp  = pct(0)
    s_dot = (f'<div class="ci-dot ci-spread" style="left:{sp:.1f}%"></div>'
             if sp is not None else "")
    leg = "● model  ◆ spread" if sp else "● model"
    return f"""
    <div class="ci-track">
      <div class="ci-fill" style="left:{bp:.1f}%;width:{bw:.1f}%"></div>
      <div style="position:absolute;top:-5px;left:{zp:.1f}%;width:2px;height:15px;
                  background:#334155;transform:translateX(-50%)"></div>
      <div class="ci-dot ci-model" style="left:{mp:.1f}%"></div>
      {s_dot}
    </div>
    <div class="ci-lbls">
      <span>{ci_low:+.1f}</span>
      <span style="color:#1e2d3d">90% CI  {leg}</span>
      <span>{ci_high:+.1f}</span>
    </div>"""

def render_card(g):
    sig  = g["signal"]; meta = SIGNAL_META[sig]
    m    = g["model_margin"]
    ci   = g["margin_90ci"]
    edge = g.get("edge")
    fav  = g["home_team_name"] if m >= 0 else g["away_team_name"]
    mcls = "pos" if m >= 0 else "neg"
    sign = "+" if m >= 0 else ""
    e_str = f"{edge*100:+.1f}%" if edge is not None else "N/A"
    # book_line is in same units as model_margin (positive = home wins)
    # e.g. book spread -4.5 → book_line +4.5 → shows as "+4.5" matching model_margin units
    bl    = g.get("book_line")
    s_str = f"{bl:+.1f}" if bl is not None else "N/A"
    ats   = ("HOME" if m > (bl or 0) else "AWAY") if bl is not None else "—"
    r_str = f"{g['rest_days']}d"

    st.markdown(f'<div class="card {sig}">', unsafe_allow_html=True)
    col_left, col_right = st.columns([3.2, 1])

    with col_left:
        st.markdown(
            f'<span class="badge {sig}">{meta["label"]}</span>'
            f'<span style="font-family:Space Mono,monospace;font-size:.68rem;'
            f'color:#334155;margin-left:10px;">{g.get("game_status","")}</span>'
            f'<div class="matchup"><span class="a">{g["away_team_name"]}</span>'
            f'<span style="color:#1e2d3d"> @ </span>'
            f'<span class="h">{g["home_team_name"]}</span></div>',
            unsafe_allow_html=True)
        st.markdown(
            f'<div class="margin-num {mcls}">{sign}{m:.1f}</div>'
            f'<div class="margin-sub">{fav} favoured by model</div>',
            unsafe_allow_html=True)
        st.markdown(ci_bar_html(ci[0], ci[1], m, g.get("home_spread")), unsafe_allow_html=True)
        ecls = "g" if (edge or 0) > 0.03 else ("r" if (edge or 0) < -0.03 else "")
        st.markdown(
            f'<div class="statrow">'
            f'<div class="statbox"><div class="statbox-val {ecls}">{e_str}</div>'
            f'<div class="statbox-lbl">Edge vs vig</div></div>'
            f'<div class="statbox"><div class="statbox-val">{s_str}</div>'
            f'<div class="statbox-lbl">Book margin</div></div>'
            f'<div class="statbox"><div class="statbox-val">{g["home_win_prob"]*100:.0f}%</div>'
            f'<div class="statbox-lbl">Home win%</div></div>'
            f'<div class="statbox"><div class="statbox-val">{ats}</div>'
            f'<div class="statbox-lbl">ATS lean</div></div>'
            f'<div class="statbox"><div class="statbox-val">{r_str}</div>'
            f'<div class="statbox-lbl">Rest days</div></div>'
            f'</div>', unsafe_allow_html=True)

        if g.get("home_ml") is not None:
            ih  = american_to_implied(g["home_ml"])
            ia  = american_to_implied(g["away_ml"]) if g.get("away_ml") else 0.5
            hec = "g" if g["home_win_prob"]-ih > 0.03 else ("r" if g["home_win_prob"]-ih < -0.03 else "")
            aec = "g" if g["away_win_prob"]-ia > 0.03 else ("r" if g["away_win_prob"]-ia < -0.03 else "")
            st.markdown(
                f'<div class="oddrow"><span class="oddlbl">ML</span>'
                f'<span class="chip {hec}">{g["home_team_abbr"]} {g["home_ml"]:+d} '
                f'impl {ih*100:.0f}% / model {g["home_win_prob"]*100:.0f}%</span>'
                f'<span class="chip {aec}">{g["away_team_abbr"]} {g["away_ml"]:+d} '
                f'impl {ia*100:.0f}% / model {g["away_win_prob"]*100:.0f}%</span>'
                f'</div>', unsafe_allow_html=True)

    with col_right:
        st.plotly_chart(donut(g["home_win_prob"], g["home_team_abbr"], g["away_team_abbr"]),
                        use_container_width=True, config={"displayModeBar": False})

    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🏀 WNBA Edge Finder")
    st.caption("XGBoost · Monte Carlo · Isotonic Calibration")
    st.divider()

    try:
        default_key = st.secrets["ODDS_API_KEY"]
        key_source  = "secrets"
    except Exception:
        default_key = os.environ.get("ODDS_API_KEY", "")
        key_source  = "env" if default_key else "manual"

    if key_source == "secrets":
        st.success("Odds API key loaded from Streamlit secrets ✓")
        api_key = default_key
    else:
        api_key = st.text_input(
            "Odds API key", value=default_key, type="password",
            help="Free at the-odds-api.com. Set via Streamlit secrets for cloud deploy.")

    st.divider()
    st.markdown("**Date**")
    game_date = st.date_input("Game date", datetime.today())
    date_str  = game_date.strftime("%Y-%m-%d")

    st.divider()
    st.markdown("**Filters**")
    min_sig      = st.selectbox("Minimum signal", SIGNAL_ORDER, index=2)
    show_no_odds = st.toggle("Show games with no odds", value=True)

    st.divider()
    st.markdown("**Model**")
    model_path = st.text_input("Predictor file", MODEL_PATH)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    f'<div class="app-title">WNBA EDGE FINDER</div>'
    f'<div class="app-sub">{date_str} &nbsp;·&nbsp; signal: {min_sig}+ &nbsp;·&nbsp; '
    f'updated: {datetime.now().strftime("%H:%M")}</div>',
    unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Load model
if not os.path.exists(model_path):
    st.error(
        f"**Model file not found:** `{model_path}`\n\n"
        "Run `python wnba_model.py` locally first, then commit "
        "`wnba_predictor_v3.joblib` to your GitHub repo.")
    st.stop()

@st.cache_resource
def _load(p): return joblib.load(p)
predictor = _load(model_path)

# Session state
if "results" not in st.session_state:
    st.session_state.results = None
if "schedule_errors" not in st.session_state:
    st.session_state.schedule_errors = []

# Run button
col_btn, col_hint = st.columns([1, 4])
with col_btn:
    run_clicked = st.button("▶  Run Analysis", type="primary", use_container_width=True)
with col_hint:
    st.markdown(
        '<p class="run-hint">Fetches schedule → team stats → player logs → '
        'live odds → runs model predictions.</p>',
        unsafe_allow_html=True)

if run_clicked:
    if not api_key:
        st.warning("No Odds API key — predictions will run without spread edge scores.")
    st.session_state.results = run_pipeline(api_key or "", date_str, predictor)

# Results
if st.session_state.results is None:
    st.markdown(
        '<div class="landing"><p>Press <b>▶ Run Analysis</b> to fetch today\'s schedule, '
        'pull live odds, and run the prediction model across all WNBA games.</p></div>',
        unsafe_allow_html=True)

elif not st.session_state.results:
    st.markdown(
        f'<div class="no-games">No WNBA games found for {date_str}.</div>',
        unsafe_allow_html=True)

    errors = st.session_state.get("schedule_errors", [])
    if errors:
        with st.expander("Why no games were found (debug)"):
            for e in errors:
                st.write(f"• {e}")

    st.markdown("---")
    st.markdown("**Enter games manually** if the schedule API is unavailable:")
    with st.form("manual_games"):
        st.caption("One game per line: Away Team @ Home Team")
        raw          = st.text_area("Games", placeholder="Indiana Fever @ Connecticut Sun\nNew York Liberty @ Washington Mystics")
        spread_input = st.text_input("Home spreads (comma-separated, same order)", placeholder="-5.5, -3.0")
        submitted    = st.form_submit_button("▶  Run with these games")

    if submitted and raw.strip():
        spreads = []
        for x in spread_input.split(","):
            try: spreads.append(float(x.strip()))
            except: spreads.append(None)
        manual_games = []
        for i, line in enumerate(raw.strip().splitlines()):
            if "@" not in line: continue
            parts     = line.split("@")
            away_name = parts[0].strip()
            home_name = parts[1].strip()
            manual_games.append({
                "game_id"        : f"manual_{i}",
                "game_status"    : "Scheduled",
                "home_team_id"   : home_name.replace(" ", "_"),
                "home_team_name" : home_name,
                "home_team_abbr" : home_name.split()[-1][:3].upper(),
                "away_team_id"   : away_name.replace(" ", "_"),
                "away_team_name" : away_name,
                "away_team_abbr" : away_name.split()[-1][:3].upper(),
            })
        if manual_games:
            season = str(game_date.year)
            with st.spinner("Loading stats and running predictions..."):
                team_df   = fetch_team_logs(season)
                player_df = fetch_player_logs(season)
                elo_map   = compute_elo(team_df) if not team_df.empty else {}
                team_df   = derive_stats(team_df) if not team_df.empty else team_df
                opp = (team_df[["GAME_ID","TEAM_ID","PTS","DREB"]]
                       .rename(columns={"TEAM_ID":"OPP_TEAM_ID","PTS":"OPP_PTS","DREB":"OPP_DREB"}))
                tdf = team_df.merge(opp, on="GAME_ID", suffixes=("","_y"))
                tdf = tdf[tdf["TEAM_ID"] != tdf["OPP_TEAM_ID"]].copy()
                if "OREB" in tdf.columns and "OPP_DREB" in tdf.columns:
                    tdf["OREB_PCT"] = tdf["OREB"] / (tdf["OREB"]+tdf["OPP_DREB"]).replace(0, np.nan)
                if "PTS" in tdf.columns:
                    tdf["MARGIN"] = tdf["PTS"] - tdf["OPP_PTS"]

                results = []
                fc = predictor.feature_cols
                for i, g in enumerate(manual_games):
                    hs       = spreads[i] if i < len(spreads) else None
                    home_elo = elo_map.get(str(g["home_team_id"]), float(ELO_BASE))
                    away_elo = elo_map.get(str(g["away_team_id"]), float(ELO_BASE))
                    home_fv  = build_fv(g["home_team_id"], g["home_team_name"], g["away_team_name"],
                                        1, tdf, player_df, elo_map, away_elo, fc)
                    away_fv  = build_fv(g["away_team_id"], g["away_team_name"], g["home_team_name"],
                                        0, tdf, player_df, elo_map, home_elo, fc)
                    book_line = -hs if hs is not None else None
                    hp = predictor.predict(home_fv, book_spread=book_line)
                    ap = predictor.predict(away_fv, book_spread=(-book_line if book_line is not None else None))
                    model_margin = (hp["predicted_margin"] - ap["predicted_margin"]) / 2
                    ci    = hp["margin_90ci"]
                    edge  = hp.get("edge_vs_spread")
                    signal = compute_signal(model_margin, book_line, edge, ci[0], ci[1])
                    results.append({
                        **g,
                        "model_margin"  : round(model_margin, 2),
                        "home_win_prob" : hp["win_prob_calibrated"],
                        "away_win_prob" : ap["win_prob_calibrated"],
                        "margin_90ci"   : ci,
                        "edge"          : edge,
                        "home_spread"   : hs,
                        "book_line"     : -hs if hs is not None else None,
                        "home_ml"       : None,
                        "away_ml"       : None,
                        "signal"        : signal,
                        "has_odds"      : hs is not None,
                        "rest_days"     : 2,
                    })
                results.sort(key=lambda x: SIGNAL_ORDER.index(x["signal"]))
                st.session_state.results = results
            st.rerun()

else:
    results  = st.session_state.results
    min_idx  = SIGNAL_ORDER.index(min_sig)
    filtered = [r for r in results
                if SIGNAL_ORDER.index(r["signal"]) <= min_idx
                and (show_no_odds or r["has_odds"])]

    n_val = sum(1 for r in results if SIGNAL_ORDER.index(r["signal"]) <= SIGNAL_ORDER.index("VALUE"))
    n_str = sum(1 for r in results if SIGNAL_ORDER.index(r["signal"]) <= SIGNAL_ORDER.index("STRONG"))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Games",          len(results))
    m2.metric("Value+ signals", n_val)
    m3.metric("Strong+",        n_str)
    m4.metric("Odds loaded",    sum(1 for r in results if r["has_odds"]))
    st.markdown("<br>", unsafe_allow_html=True)

    if not filtered:
        st.markdown(
            f'<div class="no-games">No games meet the <b>{min_sig}</b> filter today.</div>',
            unsafe_allow_html=True)
    else:
        for level in SIGNAL_ORDER:
            tier_games = [r for r in filtered if r["signal"] == level]
            if not tier_games: continue
            meta = SIGNAL_META[level]
            st.markdown(
                f'<div class="tier-lbl" style="border-top:1px solid #1e2d3d;padding-top:12px;">'
                f'{meta["label"]} · {len(tier_games)} game{"s" if len(tier_games)>1 else ""}'
                f'</div>', unsafe_allow_html=True)
            for g in tier_games:
                render_card(g)

    with st.expander("Signal guide"):
        st.markdown("""
| Level | Margin edge | Edge vs vig | CI condition | Action |
|---|---|---|---|---|
| ⚡ HIGH | > 7 pts | > 9% | Spread outside CI | 2–3 unit bet |
| ◆ STRONG | 5–7 pts | 6–9% | Spread outside CI | 1.5–2 unit bet |
| ▲ VALUE | 3–5 pts | 4–6% | Spread outside CI | 1 unit bet |
| ~ MARGINAL | 2–3 pts | 2–4% | Any | Paper trade only |
| — NOISE | < 2 pts | < 2% | Any | Skip |
        """)

    with st.expander("Raw prediction table"):
        rows = [{
            "Matchup"     : f"{r['away_team_name']} @ {r['home_team_name']}",
            "Signal"      : r["signal"],
            "Model margin": f"{r['model_margin']:+.1f}",
            "Book margin" : f"{r['book_line']:+.1f}" if r.get("book_line") is not None else "—",
            "Book spread" : f"{r['home_spread']:+.1f}" if r.get("home_spread") else "—",
            "Edge vs vig" : f"{r['edge']*100:+.1f}%" if r.get("edge") else "—",
            "Home win%"   : f"{r['home_win_prob']*100:.1f}%",
            "Away win%"   : f"{r['away_win_prob']*100:.1f}%",
            "90% CI"      : f"{r['margin_90ci'][0]:+.1f} / {r['margin_90ci'][1]:+.1f}",
        } for r in results]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
