import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show

df = pd.read_excel('datasets/excel/banki.xlsx')

X = df.drop("Nazwa", axis=1)
X = X.drop("Class", axis=1)

y = df["Class"]
sc = MinMaxScaler(feature_range=(0, 1))
sc.fit(X)
X = sc.transform(X)

print(X)
print("-----------------")
print(y)
print("-----------------")

som = MiniSom(x=25, y=25, input_len=2, sigma=1, learning_rate=0.25)
# som.random_weights_init(X)
som.pca_weights_init(X)
som.train_random(X, num_iteration=1000)

bone()
pcolor(som.distance_map().T)  # distance map as background
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    print(x)
    w = som.winner(x)
    # w[0], w[1] will place the marker at bottom left corner of the rectangle.
    # Let us add 0.5 to both of these to plot the market at the center of the rectange.
    plot(w[0] + 0.5,
         w[1] + 0.5,
         # Target value 0 will have marker "o" with color "r"
         # Target value 1 will have marker "s" with color "g"
         markers[0],
         markeredgecolor=colors[0],
         markerfacecolor='None',  # No color fill inside markers
         markersize=10,
         markeredgewidth=2)
show()

# mappings = som.win_map(X)
# print(mappings)
