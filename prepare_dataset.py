import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

dataset = pd.read_csv('FullShotsData.csv')
dataset2 = dataset[['player_id','situation', 'X', 'Y','shotType', 'h_team','a_team', 'year','xG']]


unique_situation = dataset2['situation'].unique()
situation_mapping = {situation: index for index, situation in enumerate(unique_situation)}
dataset2.loc[:, 'situation'] = dataset2['situation'].map(situation_mapping)  

unique_shotType = dataset2['shotType'].unique()
shotType_mapping = {shotType: index for index, shotType in enumerate(unique_shotType)}
dataset2.loc[:, 'shotType'] = dataset2['shotType'].map(shotType_mapping)

unique_h_team = dataset2['h_team'].unique()
h_team_mapping = {h_team: index for index, h_team in enumerate(unique_h_team)}
dataset2.loc[:, 'h_team'] = dataset2['h_team'].map(h_team_mapping)

unique_a_team = dataset2['a_team'].unique()
a_team_mapping = {a_team: index for index, a_team in enumerate(unique_a_team)}
dataset2.loc[:, 'a_team'] = dataset2['a_team'].map(a_team_mapping) 

# print(dataset2) numeric dataset

X = dataset2.iloc[:, 0:8]
y = dataset2.iloc[:, -1].round(1)  #rounding xG to 1 decimal place

# print(y)

y_names = {0.0 : '0.0', 0.1 : '0.1', 0.2 : '0.2', 0.3 : '0.3', 0.4 : '0.4', 0.5 : '0.5', 0.6 : '0.6', 0.7 : '0.7', 0.8 : '0.8', 0.9 : '0.9', 1.0 : '1.0'}
y = y.map(y_names)                #changing y to string

scaler_minmax = MinMaxScaler()
X = scaler_minmax.fit_transform(X)
# print(X) X is scaled max 1 min 0

#PCA
pca_data = PCA().fit(X)
cumulative_variance = np.cumsum(pca_data.explained_variance_ratio_)
# print(cumulative_variance)  #shows how much information is in each column
n_components_95 = np.argmax(cumulative_variance >= 0.94) + 1 #shows how many columns we need to have 95% of information
information_loss = 1 - np.sum(pca_data.explained_variance_ratio_[:n_components_95]) #shows how much information we lost
# print(n_components_95)
# print(information_loss)
X = PCA(n_components=n_components_95).fit_transform(X)
# print(X) #PCA dataset


def return_data():
    return X, y


