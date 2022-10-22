import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.decomposition import PCA


# shape = (2, 2)
# # matrix = np.zeros(shape)
# #
# # print(matrix.flatten())

def apply_scalers(df, columns_to_exclude=None):
    if columns_to_exclude:
        exclude_filter = ~df.columns.isin(columns_to_exclude)
    else:
        exclude_filter = ~df.columns.isin([])
    for column in df.iloc[:, exclude_filter].columns:
        df[column] = df[column].astype(float)

    # df.loc[:, exclude_filter] = StandardScaler().fit_transform(df.loc[:, exclude_filter])
    df.loc[:, exclude_filter] = MinMaxScaler(feature_range=(-1, 1)).fit_transform(df.loc[:, exclude_filter])
    return df


df = pd.read_excel('datasets/excel/banki.xlsx')
df.sort_values(by='Placówki', ascending=False).head()

df = apply_scalers(df, columns_to_exclude=['Nazwa'])
df.sort_values(by='Placówki', ascending=False).head()
print(df)
print("-----------------")

# kolumny do wykluczenia (te na których nie chcemy PCA)
exclude_filter = ~df.columns.isin(['Nazwa'])
# liczba głównych składowych
pca = PCA(n_components=2)
# przeliczenie
principal_components = pca.fit_transform(df.loc[:, exclude_filter])

principal_df = pd.DataFrame(data=principal_components, columns=['Placówki', 'Etaty'])

principal_df['Nazwa'] = df['Nazwa']
print(principal_df)

for x, y, name in zip(principal_df['Placówki'], principal_df['Etaty'], principal_df['Nazwa']):
    plt.annotate(name,  # this is the text
                 (x, y),  # these are the coordinates to position the label
                 textcoords="offset points",  # how to position the text
                 xytext=(0, 10),  # distance from text to points (x,y)
                 ha='center')  # horizontal alignment can be left, right or center

plt.scatter(principal_df['Placówki'], principal_df['Etaty'])
plt.show()
