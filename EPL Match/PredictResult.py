import pandas as pd
import numpy as np
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

df['MatchDate'] = pd.to_datetime(df['MatchDate'])
df['MatchYear'] = df['MatchDate'].dt.year
df['MatchMonth'] = df['MatchDate'].dt.month
df['SeasonYear'] = df['Season'].apply(lambda x: int(x.split('/')[0]))


# Encode teams
team_le = LabelEncoder()
df['HomeTeamEncoded'] = team_le.fit_transform(df['HomeTeam'])
df['AwayTeamEncoded'] = team_le.transform(df['AwayTeam'])

# Encode results
result_le = LabelEncoder()
df['target'] = result_le.fit_transform(df['FullTimeResult'])
df['HalfTimeResultEncoded'] = result_le.fit_transform(df['HalfTimeResult'])


rf= RandomForestClassifier(random_state=42, class_weight= 'balanced')

features= ['HomeShots', 'AwayShots', 'HomeTeamEncoded', 'AwayTeamEncoded',
        'HomeShotsOnTarget', 'AwayShotsOnTarget', 'HomeCorners',
        'AwayCorners', 'SeasonYear', 'MatchYear', 'MatchMonth']

def rolling_average(group, cols, new_cols):
        group= group.sort_values('MatchDate')
        rolling_stats= group[cols].rolling(12,closed='left').mean()
        group[new_cols]= rolling_stats
        group= group.dropna(subset= new_cols)
        return group

def rolling_result_counts(group, cols, new_cols):
    group = group.sort_values('MatchDate')
    rolling_counts = group[cols].rolling(12, closed='left').sum()
    group[new_cols] = rolling_counts
    return group

df['HomeWin'] = (df['FullTimeResult'] == 'H').astype(int)
df['Draw'] = (df['FullTimeResult'] == 'D').astype(int)
df['AwayWin'] = (df['FullTimeResult'] == 'A').astype(int)


stat_cols= ['HomeShots', 'AwayShots', 'HomeTeamEncoded', 'AwayTeamEncoded',
        'HomeShotsOnTarget', 'AwayShotsOnTarget', 'HomeCorners',
        'AwayCorners']
stat_new_cols= [f"{c}_rolling" for c in stat_cols]

result_cols = ['HomeWin', 'Draw', 'AwayWin']
result_new_cols = [f"{c}_rolling_count" for c in result_cols]

stat_rolling= df.groupby("HomeTeam", ).apply(lambda x: rolling_average(x,stat_cols, stat_new_cols))
stat_rolling= stat_rolling.droplevel('HomeTeam')
stat_rolling.index= range(stat_rolling.shape[0])

rolling_results = df.groupby('HomeTeam').apply(lambda x: rolling_result_counts(x, result_cols, result_new_cols))
rolling_results = rolling_results.droplevel('HomeTeam')
rolling_results.index = range(rolling_results.shape[0])

combined_df = stat_rolling.copy()
print(combined_df)

for col in result_new_cols:
    combined_df[col] = rolling_results[col]


def predict (data, predictors):
        train_seasons = [f"{year}/{str(year + 1)[-2:]}" for year in range(2005, 2021)]
        test_seasons = [f"{year}/{str(year + 1)[-2:]}" for year in range(2021, 2025)]
        train = data[data['Season'].isin(train_seasons)]
        test = data[data['Season'].isin(test_seasons)]
        rf.fit(train[predictors], train['target'])
        pred = rf.predict(test[predictors])
        accuracy = accuracy_score(test['target'], pred)
        print("Accuracy Score:", accuracy)
        cm = confusion_matrix(test['target'], pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=result_le.classes_, yticklabels=result_le.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        importances = rf.feature_importances_
        feature_names = features + stat_new_cols + result_new_cols
        feat_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=feat_importances, y=feat_importances.index)
        plt.title("Feature Importance")
        plt.xlabel("Importance Score")
        plt.show()
        print(classification_report(test['target'], pred, target_names=result_le.classes_))
        return accuracy

acc= predict(combined_df, features + stat_new_cols + result_new_cols)

