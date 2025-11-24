import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

#Austin Reaves Stats Data
data = {
    'Season': ['2021-22', '2022-23', '2023-24', '2024-25'],
    'PPG': [7.3, 13.0, 15.9, 20.2],
    'RPG': [3.2, 3.0, 4.3, 4.5],
    'APG': [1.8, 3.4, 5.5, 5.8],
    'MPG': [23.2, 28.8, 32.1, 34.9],
    'FG%': [45.9, 52.9, 48.6, 46.0],
    '3P%': [31.7, 39.8, 36.7, 37.7],
    'FT%': [83.9, 86.4, 85.3, 87.7],
}

#Austin Reaves DataFrame
df = pd.DataFrame(data)
df

df_corr = df[['PPG', 'MPG', 'APG', 'RPG', 'FG%', '3P%', 'FT%']].corr()
print(df_corr)


#Predicted Stats for 2025-2026 Season
X = df[['MPG', 'APG', 'RPG', 'FG%', '3P%', 'FT%']]
y = df['PPG']

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

#Austin Reaves Scoring Trend Plot
plt.plot(df['Season'], df['PPG'])
plt.title('Austin Reaves PPG Trend (2021–2025)')
plt.xlabel('Season')
plt.ylabel('PPG')
plt.show()


#Minus vs Points Scatter Plot
plt.scatter(df['MPG'], df['PPG'])
plt.title('Relationship Between Minutes Played and Points')
plt.xlabel('Minutes per Game')
plt.ylabel('Points per Game')
plt.show()


#Correlation Heatmap
plt.imshow(df_corr, cmap='viridis')
plt.colorbar()
plt.xticks(range(len(df_corr.columns)), df_corr.columns, rotation=45)
plt.yticks(range(len(df_corr.columns)), df_corr.columns)
plt.title('Correlation Heatmap')
plt.show()
