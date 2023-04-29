import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

dfs = pd.read_csv("data.csv") 

y = dfs['classes']
# print(y)

plot_feat = ['variance','skewness','kurtosis','entropy']

X = dfs[plot_feat]


print(X)

X_norm = (X - X.min())/(X.max() - X.min())

data_norm = pd.concat([X_norm[plot_feat], y], axis=1)

parallel_coordinates(data_norm, 'classes')
plt.show()