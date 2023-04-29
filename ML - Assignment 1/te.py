import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dfs = pd.read_csv("dataset_1.csv",header = None)
training_data = []
classes = np.zeros(1000)
weights = np.array([1,1,1],dtype=float) #including the bias as a weight
print(weights)
col = [] #Used to colour the points in the plots

def graph(training_data,weights,i):
	x = [i[0] for i in training_data]
	y = [i[1] for i in training_data]
	x = np.asarray(x)
	y = np.asarray(y)
	# print(weights)
	plt.scatter(x,y,c = col)
	plt.plot(x,(-(weights[0]/weights[2])/(weights[0]/weights[1]))*x + (-weights[0]/weights[2]),color = "Orange") #Plots the decision boundary
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.title('Dataset_1')
	# plt.show()
	plt.savefig("./images/{}.png".format(i))
	plt.close()
		
def evaluate(input_data):
	sum = np.dot(input_data,weights[1:]) + weights[0]   #Taking the dot product of the weight and input vector and adding the bias
	if sum > 0:		#Determining the target variable based on the value of the activation function
		t = 1
	else:
		t = 0
	return t
	

def learn(training_data,classes,weights,iterations = 100,learning_rate = 0.1):
	for i in range(iterations):
		print(weights)
		for data,cls in zip(training_data,classes):
			# print(data,cls,"iteration:",i)
			result = evaluate(data)
			weights[1:] += learning_rate*(cls - result)*data #Updating the weight only for misclassified patterns
			weights[0] += learning_rate*(cls - result)   #Updating the weight of bias separately as we haven't considered bias in the input data
		graph(training_data,weights,i)		
								
for i in range(len(dfs)):
	training_data.append(np.array([dfs[1][i],dfs[2][i]]))
	classes[i] = dfs[3][i]
	if classes[i] == 1:			#Setting the color for classifying the points in the plot
		col.append('b')
	else:
		col.append('r')
	
	
# print(classes)
learn(training_data,classes,weights)
# print(len(dfs))