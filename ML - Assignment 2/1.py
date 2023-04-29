import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


def sigmoid(x):
	return 1/(1+np.exp(-x))
	

def error(y,t):						#Error function for a vanilla Logistic Regression
    return (-t* np.log(y) - (1 - t) * np.log(1 - y)).mean()

	
def error_reg(y,t,lamb): 			#Error function for regression with regularisation
	return (-t* np.log(y) - (1 - t) * np.log(1 - y)).mean() + sum(weights**2)*lamb**2/(2*len(t))
	
def predict(training_data, weights, threshold=0.5):
    return sigmoid(np.dot(training_data, weights)) >= threshold
	



###########################################Logistic Regression#########################################################

def train_logistic(training_data,classes,weights,num_iter=5000,learning_rate = 0.05):
	for i in range(num_iter):
		z = np.dot(training_data,weights)
		y = sigmoid(z)
		gradient = np.dot(np.transpose(training_data),(y-classes))/len(classes)
		weights -= learning_rate*gradient 		 #Update the weights using the gradient
		# print(weights)
		if(i == num_iter-1):
			z = np.dot(training_data,weights)
			y = sigmoid(z)
			print("Error:",error(y,classes))
		# print("Error:",error(y,classes))

############################################ Logistic with L2 Regularisation ##########################################


def train_L2(training_data,classes,weights,test_data,test_classes,num_iter=5000,learning_rate = 0.05,lamb = 0.03):
	lambd = [0.0001,0.001,0.005,0.01,0.03,0.05,0.1]
	error = []
	predictions = []
	for i in range(len(lambd)):
		print("Lambda:",lambd[i])
		lamb = lambd[i]
		for i in range(num_iter):
			z = np.dot(training_data,weights)
			y = sigmoid(z)
			gradient = np.dot(np.transpose(training_data),(y-classes))/len(classes)  #Calculating Gradient
			gradient[1:] += lamb*weights[1:]/len(classes) 
			weights -= learning_rate*gradient    #Update the weights using the gradient
			
			if(i == num_iter-1):    #On the last iteration
				z = np.dot(training_data,weights)
				y = sigmoid(z)
				e = error_reg(y,classes,lamb)
				print("Error:",e)
				error.append(e)
				preds = predict(test_data,weights)  #Calculating the prediction accuracy
				accuracy = (preds == test_classes).mean()
				print("Accuracy:",accuracy)
				predictions.append(accuracy)
				print("Weights:",weights)

	return error,lambd,predictions

###################################Main Part where reading of data and calling of functions is done ###################################


dfs = pd.read_csv("data.csv",header = None) 
training_data = []
test_data = []
print(len(dfs))
classes = np.zeros(int(0.8*len(dfs)))

test_classes = []
j = 0

for i in range(0,int(0.8*len(dfs))):
	training_data.append(np.array([1,dfs[0][i],dfs[1][i],dfs[2][i],dfs[3][i]])) # 1 is the value of the intercept
	classes[j] = dfs[4][i]
	j = j+1


for i in range(int(0.8*len(dfs)),len(dfs)):
	test_data.append(np.array([1,dfs[0][i],dfs[1][i],dfs[2][i],dfs[3][i]])) # 1 is the value of the intercept
	test_classes.append(dfs[4][i])
	

weights = np.zeros(5) # Adding one for the intercept coefficient

for i in range(5):
	weights[i] = random.uniform(0,1) #Randomising the weights

tempweights = weights.copy()

print("Calling Logistic Regression")
train_logistic(training_data,classes,weights) #Calling the logistic regression training function

print("Weights:",weights)

preds = predict(test_data,weights)

print("Accuracy:",(preds == test_classes).mean())  # calculates prediction accuracy

weights = tempweights.copy()
# print(weights)

print("Calling Regularised Regression")
e,lambd,predictions = train_L2(training_data,classes,weights,test_data,test_classes)  #Calling the regularised regression function

# plt.plot(lambd,e)  						#Plots Lanbda vs Error Graph
# plt.xlabel('Regularisation Coefficient') 

# plt.ylabel('Loss') 
# plt.title('L2 Regularisation')
# plt.show() 

# plt.plot(lambd,predictions)				#Plots Lanbda vs Prediction Accuracy Graph
# plt.xlabel('Regularisation Coefficient') 

# plt.ylabel('Accuracy') 
# plt.title('L2 Regularisation')
# plt.show() 





