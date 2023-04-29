import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dfs = pd.read_csv("dataset_" + "3.csv",header = None)
training_data = []
classes = np.zeros(1000)
weights = np.array([-0.6,0.063,1.072]) #including the bias as a weight
col = [] #Used to colour the points in the plots


for i in range(len(dfs)):
	training_data.append(np.array([dfs[1][i],dfs[2][i]]))
	classes[i] = dfs[3][i]
	if classes[i] == 1:
		col.append('b')
	else:
		col.append('r')
		

		
x = [i[0] for i in training_data]
y = [i[1] for i in training_data]
x = np.asarray(x)
y = np.asarray(y)
print(weights)
plt.scatter(x,y,c = col)
plt.plot(x,(-(weights[0]/weights[2])/(weights[0]/weights[1]))*x + (-weights[0]/weights[2]),color = "Orange")
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Dataset_1')
plt.show()