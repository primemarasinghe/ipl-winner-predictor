import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ─────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="IPL Match Winner Predictor",
    page_icon="🏏",
    layout="centered"
)

# ─────────────────────────────────────────
# Load Model & Encoders
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

# ─────────────────────────────────────────
# Helper: get win rate from history
# ─────────────────────────────────────────
def get_win_rate(team):
    h = win_history.get(team, {"wins": 0, "games": 0})
    return h["wins"] / h["games"] if h["games"] > 0 else 0.5

# ─────────────────────────────────────────
# UI — Header
# ─────────────────────────────────────────
st.title("🏏 IPL Match Winner Predictor")
st.markdown("Predict which IPL team will win based on match conditions.")
st.divider()

# ─────────────────────────────────────────
# UI — Inputs
# ─────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    team1 = st.selectbox("🔵 Team 1", teams, index=0)

with col2:
    team2_options = [t for t in teams if t != team1]
    team2 = st.selectbox("🔴 Team 2", team2_options, index=1)

venue = st.selectbox("🏟️ Venue", venues)

col3, col4 = st.columns(2)

with col3:
    toss_winner = st.radio("🪙 Toss Winner", [team1, team2])

with col4:
    toss_decision = st.radio("📋 Toss Decision", ["bat", "field"])

st.divider()

# ─────────────────────────────────────────
# Predict Button
# ─────────────────────────────────────────
if st.button("🔮 Predict Winner", use_container_width=True, type="primary"):

    if team1 == team2:
        st.error("⚠️ Please select two different teams!")
    else:
        # Feature engineering (same as training)
        toss_bat        = 1 if toss_decision == "bat" else 0
        team1_won_toss  = 1 if toss_winner == team1 else 0

        t1_win_rate     = get_win_rate(team1)
        t2_win_rate     = get_win_rate(team2)
        win_rate_diff   = t1_win_rate - t2_win_rate

        # Encode
        team1_enc = le_team.transform([team1])[0]
        team2_enc = le_team.transform([team2])[0]
        venue_enc = le_venue.transform([venue])[0]

        # Build input
        X_input = np.array([[
            team1_enc, team2_enc, venue_enc,
            toss_bat, team1_won_toss,
            t1_win_rate, t2_win_rate, win_rate_diff
        ]])

        # Predict
        prediction    = model.predict(X_input)[0]
        probabilities = model.predict_proba(X_input)[0]

        team1_prob = probabilities[1] * 100
        team2_prob = probabilities[0] * 100
        winner     = team1 if prediction == 1 else team2

        # ─────────────────────────────────────────
        # Results
        # ─────────────────────────────────────────
        st.success(f"🏆 Predicted Winner: **{winner}**")

        st.markdown("### Win Probability")

        col5, col6 = st.columns(2)
        with col5:
            st.metric(label=f"🔵 {team1}", value=f"{team1_prob:.1f}%")
            st.progress(int(team1_prob))

        with col6:
            st.metric(label=f"🔴 {team2}", value=f"{team2_prob:.1f}%")
            st.progress(int(team2_prob))

        st.divider()

        # Match summary
        st.markdown("### 📋 Match Summary")
        summary_data = {
            "Detail": ["Team 1", "Team 2", "Venue", "Toss Winner", "Toss Decision",
                        "Team 1 Historical Win Rate", "Team 2 Historical Win Rate"],
            "Value": [team1, team2, venue, toss_winner, toss_decision.capitalize(),
                      f"{t1_win_rate*100:.1f}%", f"{t2_win_rate*100:.1f}%"]
        }
        st.table(pd.DataFrame(summary_data))

# ─────────────────────────────────────────
# Footer
# ─────────────────────────────────────────
st.divider()
st.caption("Built with ❤️ using Scikit-learn & Streamlit | IPL Dataset 2008–2020")