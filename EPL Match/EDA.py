import pandas as pd
import numpy as np
from pandas.core.common import random_state
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

df= pd.read_csv('epl_final.csv', low_memory= False)

df['MatchDate'] = pd.to_datetime(df['MatchDate'], format='%Y-%m-%d')
df['SeasonYear'] = df['Season'].apply(lambda x: int(x.split('/')[0]))
df['MatchYear'] = df['MatchDate'].dt.year
df['MatchMonth'] = df['MatchDate'].dt.month

print(df.describe())
print(df.columns.tolist())

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x= "FullTimeResult")
plt.title('Match result (H: Home Wins, D: Draw, A: Away Wins)')
plt.xlabel("Full Time Result")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

num_cols= df.select_dtypes(include=[np.number])

corr_matrix= num_cols.corr()

plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot= True, cmap ='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features')
plt.show()

upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

high_corr = upper_triangle[(upper_triangle.abs() > 0.6)].stack().sort_values(ascending=False)

print(high_corr)

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
sns.histplot(df['FullTimeHomeGoals'], bins= 10, kde= True, color= 'blue', label= "Home Goals")
plt.title('Distribution of Home Team goals')
plt.xlabel("Goals")
plt.ylabel("Frequency")
plt.legend()

plt.subplot(1,2,2)
sns.histplot(df['FullTimeAwayGoals'], bins= 10, kde= True, color= 'yellow', label= "Away Goals")
plt.title('Distribution of Away Team goals')
plt.xlabel("Goals")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()


home_wins = df[df['FullTimeResult'] == 'H']['HomeTeam'].value_counts()
away_wins = df[df['FullTimeResult'] == 'A']['AwayTeam'].value_counts()
total_wins = home_wins.add(away_wins, fill_value=0).sort_values(ascending=False)

plt.figure(figsize=(12,6))
total_wins.head(10).plot(kind= 'bar')
plt.title('Top 10 Teams With Most Wins')
plt.xlabel('Team')
plt.ylabel('Number of Wins')
plt.show()

home_goals = df.groupby('HomeTeam')['FullTimeHomeGoals'].sum()
away_goals = df.groupby('AwayTeam')['FullTimeAwayGoals'].sum()
total_goals = home_goals.add(away_goals, fill_value=0).sort_values(ascending=False)

plt.figure(figsize=(12, 6))
total_goals.head(10).plot(kind='bar', color= 'orange')
plt.title('Top 10 Teams by Total Goals Scored')
plt.xlabel('Team')
plt.ylabel('Total Goals')
plt.show()

plt.figure(figsize=(8, 6))
monthly_goals = df.groupby('MatchMonth')[['FullTimeHomeGoals', 'FullTimeAwayGoals']].mean()
monthly_goals.plot(marker='o')
plt.title('Average Goals per Match by Month')
plt.xlabel('Month')
plt.ylabel('Average Goals')
plt.legend(['Home Goals', 'Away Goals'])
plt.grid(True)
plt.show()


