import streamlit as st
import joblib
import numpy as np
import time
import base64
from pathlib import Path

st.set_page_config(
    page_title="IPL Winning Predictor",
    page_icon="🏏",
    layout="wide",
)

# ─────────────────────────────────────────
# Load images as base64 at runtime
# ─────────────────────────────────────────
def img_to_b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()

try:
    BALL_B64 = img_to_b64("assets/ball.png")
    BG_B64   = img_to_b64("assets/bg.png")
except:
    BALL_B64 = ""
    BG_B64   = ""

# ─────────────────────────────────────────
# Team config
# ─────────────────────────────────────────
TEAM_CONFIG = {
    "Chennai Super Kings":         {"abbr": "CSK",  "logo": "https://upload.wikimedia.org/wikipedia/en/thumb/2/2b/Chennai_Super_Kings_Logo.svg/1280px-Chennai_Super_Kings_Logo.svg.png",  "accent": "#f59e0b"},
    "Delhi Capitals":              {"abbr": "DC",   "logo": "https://upload.wikimedia.org/wikipedia/en/thumb/2/2f/Delhi_Capitals.svg/1280px-Delhi_Capitals.svg.png",                     "accent": "#2563eb"},
    "Kings XI Punjab":             {"abbr": "PBKS", "logo": "https://upload.wikimedia.org/wikipedia/en/thumb/d/d4/Punjab_Kings_Logo.svg/960px-Punjab_Kings_Logo.svg.png",                 "accent": "#ef4444"},
    "Kolkata Knight Riders":       {"abbr": "KKR",  "logo": "https://upload.wikimedia.org/wikipedia/en/thumb/4/4c/Kolkata_Knight_Riders_Logo.svg/960px-Kolkata_Knight_Riders_Logo.svg.png","accent": "#8b5cf6"},
    "Mumbai Indians":              {"abbr": "MI",   "logo": "https://upload.wikimedia.org/wikipedia/en/thumb/c/cd/Mumbai_Indians_Logo.svg/1280px-Mumbai_Indians_Logo.svg.png",             "accent": "#1d4ed8"},
    "Rajasthan Royals":            {"abbr": "RR",   "logo": "https://upload.wikimedia.org/wikipedia/en/thumb/5/5c/This_is_the_logo_for_Rajasthan_Royals%2C_a_cricket_team_playing_in_the_Indian_Premier_League_%28IPL%29.svg/960px-This_is_the_logo_for_Rajasthan_Royals%2C_a_cricket_team_playing_in_the_Indian_Premier_League_%28IPL%29.svg.png","accent": "#ec4899"},
    "Royal Challengers Bangalore": {"abbr": "RCB",  "logo": "https://upload.wikimedia.org/wikipedia/en/thumb/d/d4/Royal_Challengers_Bengaluru_Logo.svg/960px-Royal_Challengers_Bengaluru_Logo.svg.png","accent": "#dc2626"},
    "Sunrisers Hyderabad":         {"abbr": "SRH",  "logo": "https://upload.wikimedia.org/wikipedia/en/thumb/5/51/Sunrisers_Hyderabad_Logo.svg/1280px-Sunrisers_Hyderabad_Logo.svg.png",  "accent": "#f97316"},
    "Deccan Chargers":             {"abbr": "DC2",  "logo": "https://upload.wikimedia.org/wikipedia/en/a/a6/HyderabadDeccanChargers.png",                                                 "accent": "#14b8a6"},
    "Kochi Tuskers Kerala":        {"abbr": "KTK",  "logo": "https://upload.wikimedia.org/wikipedia/en/thumb/9/96/Kochi_Tuskers_Kerala_Logo.svg/1280px-Kochi_Tuskers_Kerala_Logo.svg.png","accent": "#22c55e"},
    "Pune Warriors":               {"abbr": "PW",   "logo": "https://upload.wikimedia.org/wikipedia/en/4/4a/Pune_Warriors_India_IPL_Logo.png",                                            "accent": "#0ea5e9"},
    "Rising Pune Supergiant":      {"abbr": "RPS",  "logo": "https://upload.wikimedia.org/wikipedia/en/9/9a/Rising_Pune_Supergiant.png",                                                  "accent": "#a855f7"},
    "Gujarat Lions":               {"abbr": "GL",   "logo": "https://upload.wikimedia.org/wikipedia/en/c/c4/Gujarat_Lions.png",                                                           "accent": "#f59e0b"},
}

IPL_LOGO_URL = "https://upload.wikimedia.org/wikipedia/en/thumb/8/84/Indian_Premier_League_Official_Logo.svg/1280px-Indian_Premier_League_Official_Logo.svg.png"

# ─────────────────────────────────────────
# CSS
# ─────────────────────────────────────────
bg_style = f"background-image: url('data:image/png;base64,{BG_B64}'); background-size: cover; background-position: center top; background-attachment: fixed;" if BG_B64 else "background: #040812;"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700;800&family=Bebas+Neue&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}
html, body, [class*="css"] {{ font-family: 'Barlow', sans-serif; }}

.stApp {{
    {bg_style}
    background-repeat: no-repeat;
}}
.stApp::before {{
    content: '';
    position: fixed; inset: 0;
    background: linear-gradient(180deg, rgba(4,8,18,0.84) 0%, rgba(4,8,18,0.92) 45%, rgba(4,8,18,0.97) 100%);
    z-index: 0; pointer-events: none;
}}
.block-container {{
    position: relative; z-index: 1;
    padding-top: 1.4rem; padding-bottom: 3rem;
    max-width: 1200px;
}}

/* ── Hero ── */
.hero-card {{
    padding: 2rem 2rem 1.8rem;
    border-radius: 24px;
    background: rgba(10,16,32,0.75);
    backdrop-filter: blur(20px);
    color: #fff;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 28px 70px rgba(0,0,0,0.50);
    margin-bottom: 1.4rem;
    overflow: hidden; position: relative;
}}
.hero-card::after {{
    content: '';
    position: absolute; right: -30px; bottom: -30px;
    width: 240px; height: 240px;
    background: url('data:image/png;base64,{BALL_B64}') center/contain no-repeat;
    opacity: 0.10; pointer-events: none;
}}
.hero-brand-row {{ display: flex; align-items: center; gap: 22px; flex-wrap: wrap; }}
.ipl-logo-shell {{
    width: 110px; height: 110px; border-radius: 22px;
    background: #fff; display: flex; align-items: center; justify-content: center;
    border: 1px solid rgba(255,255,255,0.14);
    box-shadow: 0 14px 36px rgba(0,0,0,0.30); flex-shrink: 0;
}}
.ipl-logo-shell img {{ width: 88px; height: 88px; object-fit: contain; }}
.hero-copy {{ flex: 1; min-width: 0; }}
.hero-title {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 52px; letter-spacing: 0.05em; line-height: 1;
    color: #fff; margin-bottom: 0.4rem;
}}
.hero-desc {{
    font-size: 14px; color: rgba(255,255,255,0.62);
    line-height: 1.55; max-width: 56ch; margin-bottom: 1rem;
}}
.hero-stats-row {{ display: flex; gap: 8px; flex-wrap: wrap; }}
.hero-stat {{
    display: flex; align-items: center; gap: 7px;
    padding: 7px 13px; border-radius: 999px;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.09);
    font-size: 12px; color: rgba(255,255,255,0.78); font-weight: 600;
}}
.hero-stat-val {{ color: #f59e0b; }}

/* ── Section label ── */
.slabel {{
    font-size: 10px; font-weight: 700; letter-spacing: 0.12em;
    color: #475569; text-transform: uppercase; margin-bottom: 10px;
}}

/* ── Panel ── */
.panel {{
    background: rgba(10,16,32,0.75);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 22px; padding: 1.4rem;
}}

/* ── Team VS ── */
.team-vs-grid {{
    display: grid; grid-template-columns: 1fr 48px 1fr;
    gap: 12px; align-items: center; margin-bottom: 1.1rem;
}}
.team-badge {{
    border-radius: 18px; padding: 16px 12px; text-align: center;
    border: 1px solid rgba(255,255,255,0.09);
    background: rgba(255,255,255,0.04); backdrop-filter: blur(10px);
}}
.team-logo {{
    width: 96px; height: 96px; border-radius: 50%;
    object-fit: contain; background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.12);
    display: block; margin: 0 auto 8px; padding: 8px;
    box-shadow: 0 12px 28px rgba(0,0,0,0.35);
}}
.team-full-name  {{ font-size: 14px; font-weight: 700; color: #f1f5f9; }}
.team-role-label {{ font-size: 10px; color: rgba(255,255,255,0.40); margin-top: 2px; text-transform: uppercase; letter-spacing: 0.06em; }}
.accent-line {{ width: 40px; height: 3px; border-radius: 999px; margin: 8px auto 0; }}
.vs-circle {{
    width: 44px; height: 44px; border-radius: 50%;
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.10);
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 800; color: rgba(255,255,255,0.55); margin: 0 auto;
}}

/* ── Predict button ── */
.stButton > button {{
    width: 100% !important; min-height: 56px;
    background: linear-gradient(135deg, #b91c1c 0%, #dc2626 40%, #f59e0b 100%) !important;
    color: #fff !important; border-radius: 16px !important;
    font-size: 17px !important; font-weight: 700 !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    box-shadow: 0 14px 36px rgba(220,38,38,0.35) !important;
    letter-spacing: 0.03em;
}}
.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 20px 48px rgba(220,38,38,0.50) !important;
}}

/* ── Ball loader ── */
.ball-loader-overlay {{
    position: fixed; inset: 0; z-index: 9999;
    background: rgba(4,8,18,0.90); backdrop-filter: blur(10px);
    display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 24px;
}}
.ball-loader-img {{
    width: 120px; height: 120px; object-fit: contain;
    animation: spinBall 0.75s linear infinite;
    filter: drop-shadow(0 0 36px rgba(220,38,38,0.75));
}}
.ball-loader-text {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 30px; letter-spacing: 0.12em; color: #fff;
}}
.ball-loader-sub {{ font-size: 13px; color: rgba(255,255,255,0.45); margin-top: -16px; letter-spacing: 0.05em; }}
.ball-loader-dots span {{
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: #f59e0b; margin: 0 4px;
    animation: dotPulse 1.2s ease-in-out infinite;
}}
.ball-loader-dots span:nth-child(2) {{ animation-delay: 0.2s; }}
.ball-loader-dots span:nth-child(3) {{ animation-delay: 0.4s; }}

@keyframes spinBall {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}
@keyframes dotPulse {{
    0%, 80%, 100% {{ transform: scale(0.6); opacity: 0.4; }}
    40% {{ transform: scale(1.0); opacity: 1; }}
}}
@keyframes floaty {{
    0%, 100% {{ transform: translateY(0px); }}
    50% {{ transform: translateY(-7px); }}
}}

/* ── Result ── */
.result-card {{
    border-radius: 22px; padding: 1.5rem; margin-bottom: 1rem; color: #fff;
    background: rgba(10,16,32,0.80); backdrop-filter: blur(18px);
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 24px 70px rgba(0,0,0,0.45);
}}
.result-winner-row {{ display: flex; align-items: center; gap: 18px; margin-bottom: 1.3rem; }}
.result-logo {{
    width: 120px; height: 120px; border-radius: 50%; object-fit: contain;
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.18);
    padding: 12px; box-shadow: 0 14px 36px rgba(0,0,0,0.40);
    animation: floaty 3s ease-in-out infinite;
}}
.result-label {{ font-size: 10px; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: rgba(255,255,255,0.55); margin-bottom: 4px; }}
.result-name {{ font-family: 'Bebas Neue', sans-serif; font-size: 42px; letter-spacing: 0.04em; color: #fff; line-height: 1; }}
.prob-block {{ margin-bottom: 12px; }}
.prob-meta {{ display: flex; justify-content: space-between; font-size: 13px; font-weight: 600; color: rgba(255,255,255,0.72); margin-bottom: 6px; }}
.prob-bar-bg   {{ height: 9px; border-radius: 999px; background: rgba(255,255,255,0.10); }}
.prob-bar-fill {{ height: 9px; border-radius: 999px; }}

/* ── Stats ── */
.stat-row {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 10px; margin-bottom: 1rem; }}
.stat-box {{
    background: rgba(10,16,32,0.75); backdrop-filter: blur(10px);
    border-radius: 16px; padding: 14px; text-align: center;
    border: 1px solid rgba(255,255,255,0.07);
}}
.stat-box-label {{ font-size: 10px; color: #475569; margin-bottom: 5px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; }}
.stat-box-value {{ font-size: 20px; font-weight: 800; color: #f1f5f9; }}

/* ── Summary ── */
.match-summary-card {{
    border-radius: 18px; padding: 1.1rem 1.2rem;
    background: rgba(10,16,32,0.75); backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.07);
}}
.summary-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
.summary-table td {{ padding: 8px 4px; border-bottom: 1px solid rgba(255,255,255,0.06); }}
.summary-table td:first-child {{ color: #475569; font-weight: 600; }}
.summary-table td:last-child  {{ font-weight: 700; color: #f1f5f9; text-align: right; }}

/* ── Preview ── */
.preview-card {{
    border-radius: 22px; padding: 1.2rem;
    background: rgba(10,16,32,0.75); backdrop-filter: blur(18px);
    color: #fff; border: 1px solid rgba(255,255,255,0.07);
    box-shadow: 0 22px 60px rgba(0,0,0,0.35);
}}
.preview-title {{ font-size: 10px; text-transform: uppercase; letter-spacing: 0.12em; color: #475569; font-weight: 700; margin-bottom: 0.7rem; }}
.preview-value {{ font-size: 16px; font-weight: 700; line-height: 1.3; margin-bottom: 0.3rem; }}
.preview-note  {{ color: rgba(255,255,255,0.45); font-size: 12px; }}
.preview-pill-row {{ display: flex; gap: 6px; flex-wrap: wrap; margin-top: 0.9rem; }}
.preview-pill {{
    padding: 5px 10px; border-radius: 999px;
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.08);
    color: rgba(255,255,255,0.72); font-size: 11px; font-weight: 600;
}}
.stSelectbox label, .stRadio label {{ font-weight: 600 !important; font-size: 12px !important; color: #64748b !important; }}
@media (max-width:640px) {{ .hero-brand-row {{ flex-direction: column; }} .hero-title {{ font-size: 38px; }} }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Load model
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    model       = joblib.load("model/model.pkl")
    le_team     = joblib.load("model/le_team.pkl")
    le_venue    = joblib.load("model/le_venue.pkl")
    teams       = joblib.load("model/teams.pkl")
    venues      = joblib.load("model/venues.pkl")
    win_history = joblib.load("model/win_history.pkl")
    return model, le_team, le_venue, teams, venues, win_history

model, le_team, le_venue, teams, venues, win_history = load_model()

def get_win_rate(team):
    h = win_history.get(team, {"wins": 0, "games": 0})
    return h["wins"] / h["games"] if h["games"] > 0 else 0.5

def tc(name):
    return TEAM_CONFIG.get(name, {"abbr": name[:3].upper(), "logo": "", "accent": "#94a3b8"})

def team_logo_html(team, size=96):
    cfg = tc(team)
    logo = cfg.get("logo", "")
    if logo:
        return f'<img class="team-logo" src="{logo}" alt="{team}" style="width:{size}px;height:{size}px;" />'
    return f'<div class="team-logo" style="width:{size}px;height:{size}px;display:flex;align-items:center;justify-content:center;font-size:{max(size//3,12)}px;font-weight:800;color:#fff;">{cfg["abbr"]}</div>'

# ─────────────────────────────────────────
# Hero
# ─────────────────────────────────────────
st.markdown(f"""
<div class="hero-card">
    <div class="hero-brand-row">
        <div class="ipl-logo-shell">
            <img src="{IPL_LOGO_URL}" alt="IPL" />
        </div>
        <div class="hero-copy">
            <div class="hero-title">IPL Winner Predictor</div>
            <div class="hero-desc">
                Powered by a Random Forest model trained on <strong style="color:#f59e0b;">748 real IPL matches</strong>
                from 2008–2020. Predictions are based on each team's historical win rate,
                venue patterns, and toss advantage — the same factors expert analysts use before every match.
            </div>
            <div class="hero-stats-row">
                <div class="hero-stat">📊 Matches: <span class="hero-stat-val">&nbsp;748</span></div>
                <div class="hero-stat">🏏 Teams: <span class="hero-stat-val">&nbsp;13</span></div>
                <div class="hero-stat">📅 Seasons: <span class="hero-stat-val">&nbsp;2008–2020</span></div>
                <div class="hero-stat">🤖 Model: <span class="hero-stat-val">&nbsp;Random Forest</span></div>
                <div class="hero-stat">🏆 Features: <span class="hero-stat-val">&nbsp;Win Rate · Venue · Toss</span></div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Layout
# ─────────────────────────────────────────
setup_col, preview_col = st.columns([1.15, 0.85], gap="large")

with setup_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="slabel">Select teams</div>', unsafe_allow_html=True)
    tc1, tc2 = st.columns(2)
    with tc1:
        team1 = st.selectbox("Team 1", teams, index=0, label_visibility="collapsed")
    with tc2:
        team2_opts = [t for t in teams if t != team1]
        team2 = st.selectbox("Team 2", team2_opts, index=0, label_visibility="collapsed")

    t1, t2 = tc(team1), tc(team2)

    st.markdown(f"""
<div class="team-vs-grid">
    <div class="team-badge" style="border-color:{t1['accent']}33;">
        {team_logo_html(team1)}
        <div class="team-full-name">{team1}</div>
        <div class="team-role-label">Side A</div>
        <div class="accent-line" style="background:{t1['accent']};"></div>
    </div>
    <div class="vs-circle">VS</div>
    <div class="team-badge" style="border-color:{t2['accent']}33;">
        {team_logo_html(team2)}
        <div class="team-full-name">{team2}</div>
        <div class="team-role-label">Side B</div>
        <div class="accent-line" style="background:{t2['accent']};"></div>
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="slabel">Venue</div>', unsafe_allow_html=True)
    venue = st.selectbox("Venue", venues, label_visibility="collapsed")

    st.markdown('<div class="slabel" style="margin-top:1rem;">Toss winner</div>', unsafe_allow_html=True)
    toss_winner = st.radio("Toss winner", [team1, team2], horizontal=True, label_visibility="collapsed", index=0)

    st.markdown('<div class="slabel" style="margin-top:1rem;">Toss decision</div>', unsafe_allow_html=True)
    toss_decision = st.radio(
        "Toss decision", ["bat", "field"], horizontal=True,
        format_func=lambda x: "🏏 Bat first" if x == "bat" else "🥅 Field first",
        key="toss_decision", label_visibility="collapsed",
    )
    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    predict = st.button("🔮  Predict Winner", use_container_width=True, type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

with preview_col:
    st.markdown(f"""
<div class="preview-card">
    <div class="preview-title">Live match snapshot</div>
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:0.8rem;">
        {team_logo_html(team1, 80)}
        <div style="font-size:11px;color:#334155;font-weight:800;letter-spacing:0.1em;">VS</div>
        {team_logo_html(team2, 80)}
    </div>
    <div class="preview-value">{team1}<br>vs {team2}</div>
    <div class="preview-note">📍 {venue}</div>
    <div class="preview-pill-row">
        <div class="preview-pill">🪙 {toss_winner}</div>
        <div class="preview-pill">{'🏏 Bat first' if toss_decision == 'bat' else '🥅 Field first'}</div>
        <div class="preview-pill">🤖 Random Forest</div>
    </div>
    <div style="height:0.8rem"></div>
    <div class="slabel">Team form</div>
    <table class="summary-table">
        <tr><td>{t1['abbr']} win rate</td><td>{get_win_rate(team1)*100:.1f}%</td></tr>
        <tr><td>{t2['abbr']} win rate</td><td>{get_win_rate(team2)*100:.1f}%</td></tr>
        <tr><td>Win edge</td><td style="color:{'#22c55e' if get_win_rate(team1)>=get_win_rate(team2) else '#f87171'};">{'+' if get_win_rate(team1)>=get_win_rate(team2) else ''}{(get_win_rate(team1)-get_win_rate(team2))*100:.1f}%</td></tr>
    </table>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Predict with 3-second ball loader
# ─────────────────────────────────────────
if predict:
    if team1 == team2:
        st.error("Please select two different teams!")
    else:
        ball_src = f"data:image/png;base64,{BALL_B64}" if BALL_B64 else "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Cricket_ball_red_and_gold.png/200px-Cricket_ball_red_and_gold.png"

        loader = st.empty()
        loader.markdown(f"""
<div class="ball-loader-overlay">
    <img class="ball-loader-img" src="{ball_src}" alt="cricket ball" />
    <div class="ball-loader-text">Analyzing Match Data</div>
    <div class="ball-loader-sub">Running Random Forest prediction...</div>
    <div class="ball-loader-dots"><span></span><span></span><span></span></div>
</div>
""", unsafe_allow_html=True)

        toss_bat       = 1 if toss_decision == "bat" else 0
        team1_won_toss = 1 if toss_winner == team1 else 0
        t1_win_rate    = get_win_rate(team1)
        t2_win_rate    = get_win_rate(team2)
        win_rate_diff  = t1_win_rate - t2_win_rate

        team1_enc = le_team.transform([team1])[0]
        team2_enc = le_team.transform([team2])[0]
        venue_enc = le_venue.transform([venue])[0]

        X_input = np.array([[team1_enc, team2_enc, venue_enc,
                             toss_bat, team1_won_toss,
                             t1_win_rate, t2_win_rate, win_rate_diff]])

        prediction    = model.predict(X_input)[0]
        probabilities = model.predict_proba(X_input)[0]

        time.sleep(3)
        loader.empty()

        team1_prob  = probabilities[1] * 100
        team2_prob  = probabilities[0] * 100
        winner      = team1 if prediction == 1 else team2
        loser       = team2 if prediction == 1 else team1
        winner_prob = max(team1_prob, team2_prob)
        loser_prob  = min(team1_prob, team2_prob)
        wc          = tc(winner)
        lc          = tc(loser)

        st.markdown(f"""
<div class="result-card" style="border-top:4px solid {wc['accent']};">
    <div class="result-winner-row">
        {team_logo_html(winner, 120)}
        <div>
            <div class="result-label" style="color:{wc['accent']};">🏆 Predicted winner</div>
            <div class="result-name">{winner}</div>
        </div>
    </div>
    <div class="prob-block">
        <div class="prob-meta"><span>{winner}</span><span>{winner_prob:.1f}%</span></div>
        <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{winner_prob:.0f}%;background:{wc['accent']};"></div></div>
    </div>
    <div class="prob-block">
        <div class="prob-meta"><span>{loser}</span><span>{loser_prob:.1f}%</span></div>
        <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{loser_prob:.0f}%;background:{lc['accent']};opacity:0.4;"></div></div>
    </div>
</div>
""", unsafe_allow_html=True)

        edge       = t1_win_rate - t2_win_rate
        edge_color = "#22c55e" if edge >= 0 else "#f87171"
        edge_str   = f"+{edge*100:.1f}%" if edge >= 0 else f"{edge*100:.1f}%"

        st.markdown(f"""
<div class="stat-row">
    <div class="stat-box"><div class="stat-box-label">{t1['abbr']} win rate</div><div class="stat-box-value">{t1_win_rate*100:.1f}%</div></div>
    <div class="stat-box"><div class="stat-box-label">{t2['abbr']} win rate</div><div class="stat-box-value">{t2_win_rate*100:.1f}%</div></div>
    <div class="stat-box"><div class="stat-box-label">Win edge</div><div class="stat-box-value" style="color:{edge_color};">{edge_str}</div></div>
</div>
<div class="match-summary-card">
    <div class="slabel" style="margin-bottom:8px;">Match summary</div>
    <table class="summary-table">
        <tr><td>Team 1</td><td>{team1}</td></tr>
        <tr><td>Team 2</td><td>{team2}</td></tr>
        <tr><td>Venue</td><td>{venue}</td></tr>
        <tr><td>Toss winner</td><td>{toss_winner}</td></tr>
        <tr><td>Toss decision</td><td>{'Bat first' if toss_decision == 'bat' else 'Field first'}</td></tr>
        <tr><td>{t1['abbr']} win rate</td><td>{t1_win_rate*100:.1f}%</td></tr>
        <tr><td>{t2['abbr']} win rate</td><td>{t2_win_rate*100:.1f}%</td></tr>
    </table>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.caption("Built with Scikit-learn & Streamlit · IPL Dataset 2008–2020 · SLIIT Data Science Project")