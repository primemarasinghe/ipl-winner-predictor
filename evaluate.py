import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import joblib
import os

# ─────────────────────────────────────────
# 1. Load & Clean
# ─────────────────────────────────────────
matches = pd.read_csv("data/matches.csv")
matches = matches.dropna(subset=["winner"])
if "method" in matches.columns:
    matches = matches[matches["method"].isna()]

# ─────────────────────────────────────────
# 2. Better Feature Engineering
# ─────────────────────────────────────────

# Historical win rate for each team UP TO that match (no data leakage)
matches = matches.sort_values("date").reset_index(drop=True)

team1_win_rates = []
team2_win_rates = []
win_history = {}

for _, row in matches.iterrows():
    t1, t2 = row["team1"], row["team2"]

    # Get past win rates (default 0.5 if no history yet)
    t1_games = win_history.get(t1, {"wins": 0, "games": 0})
    t2_games = win_history.get(t2, {"wins": 0, "games": 0})

    t1_rate = t1_games["wins"] / t1_games["games"] if t1_games["games"] > 0 else 0.5
    t2_rate = t2_games["wins"] / t2_games["games"] if t2_games["games"] > 0 else 0.5

    team1_win_rates.append(t1_rate)
    team2_win_rates.append(t2_rate)

    # Update history AFTER recording (prevent leakage)
    winner = row["winner"]
    for team in [t1, t2]:
        if team not in win_history:
            win_history[team] = {"wins": 0, "games": 0}
        win_history[team]["games"] += 1
        if team == winner:
            win_history[team]["wins"] += 1

matches["team1_win_rate"] = team1_win_rates
matches["team2_win_rate"] = team2_win_rates
matches["win_rate_diff"] = matches["team1_win_rate"] - matches["team2_win_rate"]
matches["toss_bat"] = (matches["toss_decision"] == "bat").astype(int)
matches["team1_won_toss"] = (matches["toss_winner"] == matches["team1"]).astype(int)
matches["team1_won"] = (matches["winner"] == matches["team1"]).astype(int)

# ─────────────────────────────────────────
# 3. Encode & Prepare
# ─────────────────────────────────────────
df = matches[["team1", "team2", "venue",
              "toss_bat", "team1_won_toss",
              "team1_win_rate", "team2_win_rate",
              "win_rate_diff", "team1_won"]].copy()

le_team = LabelEncoder()
le_venue = LabelEncoder()

all_teams = pd.concat([df["team1"], df["team2"]]).unique()
le_team.fit(all_teams)

df["team1_enc"]  = le_team.transform(df["team1"])
df["team2_enc"]  = le_team.transform(df["team2"])
df["venue_enc"]  = le_venue.fit_transform(df["venue"])

feature_cols = ["team1_enc", "team2_enc", "venue_enc",
                "toss_bat", "team1_won_toss",
                "team1_win_rate", "team2_win_rate", "win_rate_diff"]

X = df[feature_cols]
y = df["team1_won"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─────────────────────────────────────────
# 4. Train Improved Model
# ─────────────────────────────────────────
print("🤖 Training improved model with historical win rates...\n")

rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cv_scores = cross_val_score(rf, X, y, cv=5)

print(f"✅ Test Accuracy      : {acc*100:.2f}%")
print(f"✅ Cross-Val Accuracy : {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Team 2 Wins", "Team 1 Wins"]))

# ─────────────────────────────────────────
# 5. Confusion Matrix Plot
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["Team 2 Wins", "Team 1 Wins"])
disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title("Confusion Matrix")

# ─────────────────────────────────────────
# 6. Feature Importance Plot
# ─────────────────────────────────────────
feature_names = ["Team 1", "Team 2", "Venue",
                 "Toss Decision", "Toss Winner",
                 "Team 1 Win Rate", "Team 2 Win Rate", "Win Rate Diff"]
importances = rf.feature_importances_
indices = np.argsort(importances)

axes[1].barh([feature_names[i] for i in indices],
             importances[indices], color="steelblue")
axes[1].set_title("Feature Importances")
axes[1].set_xlabel("Importance Score")

plt.suptitle("IPL Match Winner Predictor — Model Evaluation", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("evaluation_report.png", dpi=150)
plt.show()
print("\n✅ Evaluation chart saved: evaluation_report.png")

# ─────────────────────────────────────────
# 7. Save Improved Model
# ─────────────────────────────────────────
os.makedirs("model", exist_ok=True)
joblib.dump(rf, "model/model.pkl")
joblib.dump(le_team, "model/le_team.pkl")
joblib.dump(le_venue, "model/le_venue.pkl")
joblib.dump(sorted(list(le_team.classes_)), "model/teams.pkl")
joblib.dump(sorted(list(le_venue.classes_)), "model/venues.pkl")

# Save current win rates for use in the app
joblib.dump(win_history, "model/win_history.pkl")

print("✅ Improved model saved!")
print("\n🎉 Step 4 Complete! Ready for Step 5 — Streamlit App")