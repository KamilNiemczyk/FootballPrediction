from prepare_dataset import return_data
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# X, y = return_data()

dataset = pd.read_csv('FullShotsData.csv')
data = dataset[['player_id','situation', 'X', 'Y','shotType', 'h_team','a_team','xG']]
data = pd.DataFrame(data)
data = data.round(1)
data = data.astype(str)

data_encoded = pd.get_dummies(data)
data_encoded = data_encoded.astype(bool).astype(int)
# print(data_encoded.head())

frequent_itemsets = apriori(data_encoded, min_support=0.2, use_colnames=True)
# print(frequent_itemsets)


