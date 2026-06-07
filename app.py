import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="IPL WInning Predictor",
    page_icon="🏏",
    layout="wide",
)

TEAM_CONFIG = {
    "Chennai Super Kings": {"abbr": "CSK", "logo": "https://upload.wikimedia.org/wikipedia/en/thumb/2/2b/Chennai_Super_Kings_Logo.svg/1280px-Chennai_Super_Kings_Logo.svg.png", "accent": "#f59e0b"},
    "Delhi Capitals": {"abbr": "DC", "logo": "https://upload.wikimedia.org/wikipedia/en/thumb/2/2f/Delhi_Capitals.svg/1280px-Delhi_Capitals.svg.png", "accent": "#2563eb"},
    "Kings XI Punjab": {"abbr": "PBKS", "logo": "https://upload.wikimedia.org/wikipedia/en/thumb/d/d4/Punjab_Kings_Logo.svg/960px-Punjab_Kings_Logo.svg.png", "accent": "#ef4444"},
    "Kolkata Knight Riders": {"abbr": "KKR", "logo": "https://upload.wikimedia.org/wikipedia/en/thumb/4/4c/Kolkata_Knight_Riders_Logo.svg/960px-Kolkata_Knight_Riders_Logo.svg.png", "accent": "#8b5cf6"},
    "Mumbai Indians": {"abbr": "MI", "logo": "https://upload.wikimedia.org/wikipedia/en/thumb/c/cd/Mumbai_Indians_Logo.svg/1280px-Mumbai_Indians_Logo.svg.png", "accent": "#1d4ed8"},
    "Rajasthan Royals": {"abbr": "RR", "logo": "https://upload.wikimedia.org/wikipedia/en/thumb/5/5c/This_is_the_logo_for_Rajasthan_Royals%2C_a_cricket_team_playing_in_the_Indian_Premier_League_%28IPL%29.svg/960px-This_is_the_logo_for_Rajasthan_Royals%2C_a_cricket_team_playing_in_the_Indian_Premier_League_%28IPL%29.svg.png", "accent": "#ec4899"},
    "Royal Challengers Bangalore": {"abbr": "RCB", "logo": "https://upload.wikimedia.org/wikipedia/en/thumb/d/d4/Royal_Challengers_Bengaluru_Logo.svg/960px-Royal_Challengers_Bengaluru_Logo.svg.png", "accent": "#dc2626"},
    "Sunrisers Hyderabad": {"abbr": "SRH", "logo": "https://upload.wikimedia.org/wikipedia/en/thumb/5/51/Sunrisers_Hyderabad_Logo.svg/1280px-Sunrisers_Hyderabad_Logo.svg.png", "accent": "#f97316"},
    "Deccan Chargers": {"abbr": "DC2", "logo": "https://upload.wikimedia.org/wikipedia/en/a/a6/HyderabadDeccanChargers.png", "accent": "#14b8a6"},
    "Kochi Tuskers Kerala": {"abbr": "KTK", "logo": "https://upload.wikimedia.org/wikipedia/en/thumb/9/96/Kochi_Tuskers_Kerala_Logo.svg/1280px-Kochi_Tuskers_Kerala_Logo.svg.png", "accent": "#22c55e"},
    "Pune Warriors": {"abbr": "PW", "logo": "https://upload.wikimedia.org/wikipedia/en/4/4a/Pune_Warriors_India_IPL_Logo.png", "accent": "#0ea5e9"},
    "Rising Pune Supergiant": {"abbr": "RPS", "logo": "https://upload.wikimedia.org/wikipedia/en/9/9a/Rising_Pune_Supergiant.png", "accent": "#a855f7"},
    "Gujarat Lions": {"abbr": "GL", "logo": "https://upload.wikimedia.org/wikipedia/en/c/c4/Gujarat_Lions.png", "accent": "#f59e0b"},
}

IPL_LOGO_URL = "https://upload.wikimedia.org/wikipedia/en/thumb/8/84/Indian_Premier_League_Official_Logo.svg/1280px-Indian_Premier_League_Official_Logo.svg.png"

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
    background:
        radial-gradient(circle at top left, rgba(34, 197, 94, 0.16), transparent 24%),
        radial-gradient(circle at top right, rgba(245, 158, 11, 0.16), transparent 24%),
        linear-gradient(180deg, #07111f 0%, #0b1220 42%, #111827 100%);
}

.stApp {
    background:
        radial-gradient(circle at 18% 18%, rgba(34, 197, 94, 0.18) 0 18%, transparent 19% 100%),
        radial-gradient(circle at 82% 10%, rgba(245, 158, 11, 0.18) 0 16%, transparent 17% 100%),
        radial-gradient(circle at 70% 75%, rgba(59, 130, 246, 0.16) 0 20%, transparent 21% 100%),
        linear-gradient(180deg, #07111f 0%, #0b1220 42%, #111827 100%);
}

.block-container {
    padding-top: 1.4rem;
    padding-bottom: 2rem;
    max-width: 1180px;
}

.hero-card {
    padding: 1.4rem 1.5rem;
    border-radius: 24px;
    background:
        radial-gradient(circle at top right, rgba(34,197,94,0.12), transparent 24%),
        radial-gradient(circle at top left, rgba(245,158,11,0.12), transparent 24%),
        linear-gradient(135deg, #0b1220 0%, #0f172a 55%, #020617 100%);
    color: #fff;
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 28px 70px rgba(0, 0, 0, 0.32);
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
}

.hero-card::before {
    content: "";
    position: absolute;
    inset: 0;
    background:
        radial-gradient(circle at 20% 18%, rgba(34,197,94,0.10), transparent 18%),
        radial-gradient(circle at 80% 12%, rgba(245,158,11,0.12), transparent 16%),
        linear-gradient(180deg, transparent 0%, rgba(255,255,255,0.02) 100%);
    pointer-events: none;
}

.hero-title {
    font-size: 44px;
    font-weight: 700;
    line-height: 1.05;
    margin: 0;
}

.hero-brand-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 22px;
    flex-wrap: wrap;
}

.hero-copy {
    min-width: 0;
}

.ipl-logo-shell {
    width: 124px;
    height: 124px;
    border-radius: 28px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #ffffff;
    border: 1px solid rgba(255,255,255,0.14);
    box-shadow: 0 18px 42px rgba(0,0,0,0.22);
    position: relative;
}

.ipl-logo-shell img {
    width: 102px;
    height: 102px;
    object-fit: contain;
    animation: floaty 4s ease-in-out infinite;
}

.section-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: #64748b;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.team-vs-grid {
    display: grid;
    grid-template-columns: 1fr 48px 1fr;
    gap: 12px;
    align-items: center;
    margin-bottom: 1rem;
}

.team-badge {
    border-radius: 18px;
    padding: 18px 14px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.09);
    background: rgba(15, 23, 42, 0.68) !important;
    backdrop-filter: blur(14px);
    box-shadow: 0 18px 40px rgba(0,0,0,0.22);
}

.team-logo {
    width: 128px;
    height: 128px;
    border-radius: 50%;
    object-fit: contain;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.12);
    display: block;
    margin: 0 auto 8px;
    padding: 10px;
    box-shadow: 0 18px 36px rgba(0,0,0,0.28);
    animation: pulseGlow 3.8s ease-in-out infinite;
}

.team-full-name  { font-size: 15px; font-weight: 700; color: #fff; }
.team-role-label { font-size: 11px; color: rgba(255,255,255,0.66); margin-top: 2px; }

.vs-circle {
    width: 44px;
    height: 44px;
    border-radius: 50%;
    background: rgba(255,255,255,0.08);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    font-weight: 700;
    color: rgba(255,255,255,0.9);
    margin: 0 auto;
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.18);
}

.predict-btn-wrap button {
    width: 100% !important;
    background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%) !important;
    color: white !important;
    border-radius: 14px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    padding: 0.75rem !important;
    border: none !important;
    box-shadow: 0 12px 26px rgba(29, 78, 216, 0.25);
}

.result-card {
    border-radius: 22px;
    padding: 1.4rem;
    margin-bottom: 1rem;
    color: #fff;
    background:
        radial-gradient(circle at top right, rgba(245,158,11,0.14), transparent 24%),
        radial-gradient(circle at bottom left, rgba(34,197,94,0.10), transparent 24%),
        linear-gradient(135deg, rgba(15,23,42,0.96) 0%, rgba(2,6,23,0.98) 100%) !important;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 24px 70px rgba(0, 0, 0, 0.30);
}

.result-winner-row {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 1.2rem;
}

.result-logo {
    width: 144px;
    height: 144px;
    border-radius: 50%;
    object-fit: contain;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.22);
    padding: 14px;
    box-shadow: 0 18px 42px rgba(0,0,0,0.26);
    animation: pulseGlow 3.8s ease-in-out infinite;
}

.result-label {
    font-size: 11px;
    font-weight: 600;
    opacity: 1;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: rgba(255,255,255,0.68);
}

.result-name  { font-size: 24px; font-weight: 700; }

.prob-block { margin-bottom: 10px; }
.prob-meta  { display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 5px; font-weight: 500; }
.prob-bar-bg   { height: 9px; border-radius: 999px; background: rgba(255,255,255,0.16); }
.prob-bar-fill { height: 9px; border-radius: 999px; background: #f59e0b; }

.stat-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-top: 1rem;
}

.stat-box {
    background: rgba(15,23,42,0.72);
    border-radius: 16px;
    padding: 12px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
}

.stat-box-label { font-size: 11px; color: rgba(255,255,255,0.64); margin-bottom: 4px; }
.stat-box-value { font-size: 18px; font-weight: 700; color: #fff; }

.summary-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.summary-table td { padding: 8px 4px; border-bottom: 1px solid rgba(255,255,255,0.08); }
.summary-table td:first-child { color: rgba(255,255,255,0.62); width: 45%; }
.summary-table td:last-child  { font-weight: 600; color: #fff; }

.stSelectbox label, .stRadio label { font-weight: 600 !important; font-size: 13px !important; }

.preview-card {
    border-radius: 22px;
    padding: 1.1rem;
    background:
        radial-gradient(circle at top, rgba(59,130,246,0.10), transparent 24%),
        linear-gradient(180deg, rgba(15,23,42,0.82) 0%, rgba(2,6,23,0.92) 100%);
    color: #fff;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 22px 60px rgba(0, 0, 0, 0.24);
    position: relative;
    overflow: hidden;
}

.preview-title {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: rgba(255,255,255,0.64);
    font-weight: 700;
    margin-bottom: 0.55rem;
}

.preview-value {
    font-size: 18px;
    font-weight: 700;
    line-height: 1.2;
    margin-bottom: 0.25rem;
}

.preview-note {
    color: rgba(255,255,255,0.72);
    font-size: 13px;
}

.preview-pill-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 0.9rem;
}

.preview-pill {
    padding: 0.45rem 0.7rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.08);
    color: #fff;
    font-size: 12px;
    border: 1px solid rgba(255,255,255,0.08);
}

.match-summary-card {
    border-radius: 20px;
    padding: 1rem 1.05rem;
    background: rgba(15,23,42,0.72);
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
}

.crowd-bar {
@keyframes floaty {
    0%, 100% { transform: translateY(0px) rotate(0deg); }
    50% { transform: translateY(-6px) rotate(2deg); }
}

@keyframes pulseGlow {
    0%, 100% { transform: scale(1); box-shadow: 0 18px 36px rgba(0,0,0,0.28); }
    50% { transform: scale(1.03); box-shadow: 0 24px 44px rgba(0,0,0,0.34); }
}

.logo-wrap {
    display: flex;
    align-items: center;
    justify-content: center;
}

.team-card-top {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
}

.accent-line {
    width: 44px;
    height: 3px;
    border-radius: 999px;
    background: rgba(255,255,255,0.22);
    margin: 8px auto 0;
}

@media (max-width: 640px) {
    .hero-brand-row {
        flex-direction: column;
        text-align: center;
    }

    .hero-copy {
        min-width: 100%;
    }

    .hero-title {
        font-size: 34px;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    model = joblib.load("model/model.pkl")
    le_team = joblib.load("model/le_team.pkl")
    le_venue = joblib.load("model/le_venue.pkl")
    teams = joblib.load("model/teams.pkl")
    venues = joblib.load("model/venues.pkl")
    win_history = joblib.load("model/win_history.pkl")
    return model, le_team, le_venue, teams, venues, win_history


model, le_team, le_venue, teams, venues, win_history = load_model()


def get_win_rate(team):
    h = win_history.get(team, {"wins": 0, "games": 0})
    return h["wins"] / h["games"] if h["games"] > 0 else 0.5


def team_cfg(name):
    return TEAM_CONFIG.get(
        name,
        {"abbr": name[:3].upper(), "logo": "", "accent": "#94a3b8"},
    )


def team_logo_html(team, size=96):
    cfg = team_cfg(team)
    logo = cfg.get("logo", "")
    if logo:
        return f'<img class="team-logo" src="{logo}" alt="{team} logo" style="width:{size}px;height:{size}px;" />'
    return f'<div class="team-logo" style="width:{size}px;height:{size}px;display:flex;align-items:center;justify-content:center;font-size:{max(size//3, 12)}px;font-weight:700;color:#fff;">{cfg["abbr"]}</div>'


st.markdown(
    f"""
<div class="hero-card">
    <div class="hero-brand-row">
        <div class="ipl-logo-shell">
            <img src="{IPL_LOGO_URL}" alt="IPL logo" />
        </div>
        <div class="hero-copy">
            <div class="hero-title">IPL WInning Predictor</div>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

setup_col, preview_col = st.columns([1.15, 0.85], gap="large")

with setup_col:
    st.markdown('<div class="section-label">Match setup</div>', unsafe_allow_html=True)
    team_col1, team_col2 = st.columns(2)
    with team_col1:
        team1 = st.selectbox("Team 1", teams, index=0, label_visibility="collapsed")
    with team_col2:
        team2_opts = [t for t in teams if t != team1]
        team2 = st.selectbox("Team 2", team2_opts, index=0, label_visibility="collapsed")

    t1 = team_cfg(team1)
    t2 = team_cfg(team2)

    st.markdown(
        f"""
<div class="team-vs-grid">
    <div class="team-badge">
        <div class="team-card-top">
            {team_logo_html(team1)}
            <div class="team-full-name">{team1}</div>
            <div class="team-role-label">Selected side A</div>
        </div>
        <div class="accent-line" style="background:{t1['accent']};"></div>
    </div>
    <div class="vs-circle">VS</div>
    <div class="team-badge">
        <div class="team-card-top">
            {team_logo_html(team2)}
            <div class="team-full-name">{team2}</div>
            <div class="team-role-label">Selected side B</div>
        </div>
        <div class="accent-line" style="background:{t2['accent']};"></div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">Venue</div>', unsafe_allow_html=True)
    venue = st.selectbox("Venue", venues, label_visibility="collapsed")

    st.markdown('<div class="section-label">Toss winner</div>', unsafe_allow_html=True)
    toss_winner = st.radio(
        "Toss winner",
        [team1, team2],
        horizontal=True,
        label_visibility="collapsed",
        index=0,
    )

    st.markdown('<div class="section-label">Toss decision</div>', unsafe_allow_html=True)
    toss_decision = st.radio(
        "Toss decision",
        ["bat", "field"],
        horizontal=True,
        format_func=lambda x: "Bat first" if x == "bat" else "Field first",
        key="toss_decision",
        label_visibility="collapsed",
    )

with preview_col:
    st.markdown(
        f"""
<div class="preview-card">
    <div class="preview-title">Live match snapshot</div>
    <div style="display:flex; align-items:center; gap:14px; margin-bottom:0.8rem;">
        {team_logo_html(team1, 88)}
        <div style="font-size:12px; color:#64748b; font-weight:700; letter-spacing:0.08em;">VS</div>
        {team_logo_html(team2, 88)}
    </div>
    <div class="preview-value">{team1} vs {team2}</div>
    <div class="preview-note">{venue}</div>
    <div class="preview-pill-row">
        <div class="preview-pill">Toss winner: {toss_winner}</div>
        <div class="preview-pill">Decision: {'Bat first' if toss_decision == 'bat' else 'Field first'}</div>
        <div class="preview-pill">Model: Random Forest</div>
    </div>
    <div style='height:0.9rem;'></div>
    <div class="match-summary-card">
        <div class="section-label" style="margin-bottom:8px;">Team form snapshot</div>
        <table class="summary-table">
            <tr><td>{t1['abbr']} win rate</td><td>{get_win_rate(team1)*100:.1f}%</td></tr>
            <tr><td>{t2['abbr']} win rate</td><td>{get_win_rate(team2)*100:.1f}%</td></tr>
            <tr><td>Venue</td><td>{venue}</td></tr>
        </table>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:0.9rem;'></div>", unsafe_allow_html=True)

predict_col1, predict_col2, predict_col3 = st.columns([1, 1.4, 1])
with predict_col2:
    predict = st.button("🔮  Predict Winner", use_container_width=True, type="primary")

if predict:
    toss_bat = 1 if toss_decision == "bat" else 0
    team1_won_toss = 1 if toss_winner == team1 else 0
    t1_win_rate = get_win_rate(team1)
    t2_win_rate = get_win_rate(team2)
    win_rate_diff = t1_win_rate - t2_win_rate

    team1_enc = le_team.transform([team1])[0]
    team2_enc = le_team.transform([team2])[0]
    venue_enc = le_venue.transform([venue])[0]

    X_input = np.array([[
        team1_enc,
        team2_enc,
        venue_enc,
        toss_bat,
        team1_won_toss,
        t1_win_rate,
        t2_win_rate,
        win_rate_diff,
    ]])

    prediction = model.predict(X_input)[0]
    probabilities = model.predict_proba(X_input)[0]

    team1_prob = probabilities[1] * 100
    team2_prob = probabilities[0] * 100
    winner = team1 if prediction == 1 else team2
    loser = team2 if prediction == 1 else team1
    winner_prob = max(team1_prob, team2_prob)
    loser_prob = min(team1_prob, team2_prob)

    wc = team_cfg(winner)

    st.markdown(
        f"""
<div class="result-card" style="border-top: 4px solid {wc['accent']};">
    <div class="result-winner-row">
        {team_logo_html(winner, 156)}
        <div>
            <div class="result-label">Predicted winner</div>
            <div class="result-name">{winner}</div>
        </div>
    </div>
    <div class="prob-block">
        <div class="prob-meta"><span>{winner}</span><span>{winner_prob:.1f}%</span></div>
        <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{winner_prob:.0f}%;"></div></div>
    </div>
    <div class="prob-block">
        <div class="prob-meta"><span>{loser}</span><span>{loser_prob:.1f}%</span></div>
        <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{loser_prob:.0f}%; background:rgba(255,255,255,0.35);"></div></div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    edge = t1_win_rate - t2_win_rate
    edge_color = "#1D9E75" if edge >= 0 else "#E24B4A"
    edge_str = f"+{edge*100:.1f}%" if edge >= 0 else f"{edge*100:.1f}%"

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.markdown(
            f"""
<div class="stat-box">
    <div class="stat-box-label">{t1['abbr']} win rate</div>
    <div class="stat-box-value">{t1_win_rate*100:.1f}%</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with metric_col2:
        st.markdown(
            f"""
<div class="stat-box">
    <div class="stat-box-label">{t2['abbr']} win rate</div>
    <div class="stat-box-value">{t2_win_rate*100:.1f}%</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with metric_col3:
        st.markdown(
            f"""
<div class="stat-box">
    <div class="stat-box-label">Win rate edge</div>
    <div class="stat-box-value" style="color:{edge_color};">{edge_str}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
<div class="match-summary-card">
    <h5 style="margin-top:0; margin-bottom:0.75rem;">Match summary</h5>
<table class="summary-table">
    <tr><td>Team 1</td><td>{team1}</td></tr>
    <tr><td>Team 2</td><td>{team2}</td></tr>
    <tr><td>Venue</td><td>{venue}</td></tr>
    <tr><td>Toss winner</td><td>{toss_winner}</td></tr>
    <tr><td>Toss decision</td><td>{'Bat first' if toss_decision == 'bat' else 'Field first'}</td></tr>
    <tr><td>Team 1 win rate</td><td>{t1_win_rate*100:.1f}%</td></tr>
    <tr><td>Team 2 win rate</td><td>{t2_win_rate*100:.1f}%</td></tr>
</table>
    </div>
""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)
st.caption("Built with Scikit-learn & Streamlit · IPL Dataset 2008–2020 · SLIIT Data Science Project")
