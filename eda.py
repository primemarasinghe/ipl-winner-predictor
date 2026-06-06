import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

matches = pd.read_csv("data/matches.csv")

Path("charts").mkdir(exist_ok=True)

print("Total matches before cleaning:", len(matches))

# Basic clean - drop rows where winner is missing (no result / tied)
matches = matches.dropna(subset=["winner"])

print("Total matches after cleaning:", len(matches))
print("\nColumns:", matches.columns.tolist())

# 1. Top teams by wins
plt.figure(figsize=(12, 5)) #create a canvas for the plot
win_counts = matches["winner"].value_counts()
print("\nWin counts:\n", win_counts)
sns.barplot(x=win_counts.values, y=win_counts.index, palette="viridis") #palette="viridis" gives the bars a color gradient.
plt.title("Total Wins by Team (All Seasons)")
plt.xlabel("Wins")
plt.ylabel("Team")
plt.tight_layout() #Adjust spacing
plt.savefig("charts/eda_team_wins.png")
plt.show()
print("Chart 1 is saved: charts/eda_team_wins.png")


# 2. Toss winner vs Match winner
print("\nMatches won by toss winner:", (matches["toss_winner"] == matches["winner"]).sum())
matches["Win_toss_and_match"] = matches["toss_winner"] == matches["winner"]
toss_effect = matches["Win_toss_and_match"].value_counts()

plt.figure(figsize=(6, 5))
plt.pie(
    toss_effect.values,
    labels=["Toss Winner Lost", "Toss Winner Won"],
    autopct="%1.1f%%",
    colors=["#ff6b6b", "#51cf66"],
    startangle=90
)
plt.title("Does Winning the Toss Help Win the Match?")
plt.tight_layout()
plt.savefig("charts/eda_toss_effect.png")
plt.show()
print("Chart 2 is saved: charts/eda_toss_effect.png")

# 3. Toss decision - bat vs field
plt.figure(figsize=(6, 5))
toss_decision = matches["toss_decision"].value_counts()
sns.barplot(x=toss_decision.index, 
            y=toss_decision.values, 
            palette="coolwarm")
plt.title("Toss Decision: Bat vs Field")
plt.xlabel("Decision")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("charts/eda_toss_decision.png")
plt.show()
print("Chart 3 is saved: charts/eda_toss_decision.png")

# 4. Top venues by number of matches
plt.figure(figsize=(12, 5))
venue_counts = matches["venue"].value_counts().head(10)
sns.barplot(x=venue_counts.values, y=venue_counts.index, palette="mako")
plt.title("Top 10 Venues by Number of Matches")
plt.xlabel("Matches Played")
plt.ylabel("Venue")
plt.tight_layout()
plt.savefig("charts/eda_venues.png")
plt.show()
print("Chart 4 is saved: charts/eda_venues.png") 

#5. Wins by season
plt.figure(figsize=(12, 5))
season_matches=matches.groupby("season")["id"].count()
# marker="o" adds points to the line plot, color="royalblue" sets the line color.
sns.lineplot(x=season_matches.index, y=season_matches.values, marker="o", color="royalblue") 
plt.title("Number of Matches Per Season")
plt.xlabel("Season")
plt.ylabel("Matches")
plt.tight_layout()
plt.savefig("charts/eda_seasons.png")
plt.show()
print("Chart 5 is saved: charts/eda_seasons.png")

toss_win_pct = matches["Win_toss_and_match"].mean() * 100

print(f"\n📊 Key Insights:")
print(f"   • Toss winner also won the match: {toss_win_pct:.1f}% of the time")
print(f"   • Most successful team: {win_counts.index[0]} ({win_counts.values[0]} wins)")
print(f"   • Most used venue: {matches['venue'].value_counts().index[0]}")
print(f"   • Seasons covered: {matches['season'].min()} to {matches['season'].max()}")
