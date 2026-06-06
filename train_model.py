import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import os

# 1. Load & Clean Data
matches = pd.read_csv("data/matches.csv")
matches = matches.dropna(subset=["winner"])

# Remove Super Overs (tie breakers - not normal matches)
if "method" in matches.columns:
    matches = matches[matches["method"].isna()]# Checks where values are missing (NaN)
print(f"✅ Clean matches loaded: {len(matches)}")

# 2. Feature Engineering
# Feature 1: Did toss winner choose to bat?
matches["toss_bat"] = (matches["toss_decision"] == "bat").astype(int)

# Feature 2: Did team1 win the toss?
matches["team1_won_toss"] = (matches["toss_winner"] == matches["team1"]).astype(int)

# Feature 3: Target — did team1 win?
matches["team1_won"] = (matches["winner"] == matches["team1"]).astype(int)

# Select features
df = matches[["team1", "team2", "venue", "toss_bat", "team1_won_toss", "team1_won"]].copy()

print(f"✅ Features selected: {df.columns.tolist()}")
print(f"   Class balance — team1 won: {df['team1_won'].mean()*100:.1f}%")

# 3. Encode Categorical Columns
le_team = LabelEncoder()
le_venue = LabelEncoder()

# Fit on all teams (team1 + team2 combined so encoder knows all teams)
all_teams = pd.concat([df["team1"], df["team2"]]).unique()
le_team.fit(all_teams)

df["team1_enc"] = le_team.transform(df["team1"])
df["team2_enc"] = le_team.transform(df["team2"])
df["venue_enc"] = le_venue.fit_transform(df["venue"])

print(f"✅ Teams encoded: {list(le_team.classes_)}")
print(f"✅ Venues encoded: {list(le_venue.classes_)}")

# 4. Prepare X and y
X = df[["team1_enc", "team2_enc", "venue_enc", "toss_bat", "team1_won_toss"]]
y = df["team1_won"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n✅ Train size: {len(X_train)} | Test size: {len(X_test)}")


# 5. Train Models
print("\n🤖 Training models...")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))

# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test))

print(f"\n📊 Model Comparison:")
print(f"   Random Forest Accuracy     : {rf_acc*100:.2f}%")
print(f"   Logistic Regression Accuracy: {lr_acc*100:.2f}%")

# Pick best model
best_model = rf if rf_acc >= lr_acc else lr
best_name = "Random Forest" if rf_acc >= lr_acc else "Logistic Regression"
print(f"\n🏆 Best model: {best_name}")

# 6. Feature Importance (Random Forest)
feature_names = ["Team 1", "Team 2", "Venue", "Toss Decision", "Toss Winner"]
importances = rf.feature_importances_

print(f"\n📌 Feature Importances (Random Forest):")
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    print(f"   {name:<20}: {imp:.4f}")

# 7. Save Model & Encoders
os.makedirs("model", exist_ok=True)

joblib.dump(best_model, "model/model.pkl")
joblib.dump(le_team, "model/le_team.pkl")
joblib.dump(le_venue, "model/le_venue.pkl")

# Save team and venue lists for the app
joblib.dump(sorted(list(le_team.classes_)), "model/teams.pkl")
joblib.dump(sorted(list(le_venue.classes_)), "model/venues.pkl")

print(f"\n✅ Model saved to model/model.pkl")
print(f"✅ Encoders saved to model/")
print(f"\n🎉 Step 3 Complete! Ready for Step 4 — Evaluation")