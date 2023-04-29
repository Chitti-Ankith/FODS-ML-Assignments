import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Plots the perceptron function
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
	

def learn(training_data,classes,weights,iterations = 50,learning_rate = 0.1):
	for i in range(iterations):
		print(weights)
		for data,cls in zip(training_data,classes):
			# print(data,cls,"iteration:",i)
			result = evaluate(data)
			weights[1:] += learning_rate*(cls - result)*data #Updating the weight only for misclassified patterns
			weights[0] += learning_rate*(cls - result)   #Updating the weight of bias separately as we haven't considered bias in the input data
		# graph(training_data,weights,i)	
		#Uncomment the above line to plot the images. Need to have an images folder.	
								

#Main Function for reading the datasets
								
for i in range(3):
	dfs = pd.read_csv("dataset_" + str(i+1) + ".csv",header = None) 
	training_data = []
	classes = np.zeros(1000)
	weights = np.array([0.8,0.8,0.8]) #including the bias as a weight
	col = [] #Used to colour the points in the plots	
	# print(classes)
	for j in range(len(dfs)):
		training_data.append(np.array([dfs[1][j],dfs[2][j]]))
		classes[j] = dfs[3][j]
		if classes[j] == 1:			#Setting the color for classifying the points in the plot
			col.append('b')
		else:
			col.append('r')
	
	print("For Data Set:",i+1)
	learn(training_data,classes,weights)
	# print(len(dfs))