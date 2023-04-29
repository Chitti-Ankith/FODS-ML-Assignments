import numpy as np

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

	
def feedforward(inputs,weights1,weights2):
	layer1 = sigmoid(np.dot(inputs, weights1))
	output = sigmoid(np.dot(layer1, weights2))
	return layer1,output

def backprop(inputs,y,weights1,weights2,layer1,output):
	# application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
	d_weights2 = np.dot(layer1.T, (2*(y - output) * sigmoid_derivative(output)))
	d_weights1 = np.dot(inputs.T,  (np.dot(2*(y - output) * sigmoid_derivative(output), weights2.T) * sigmoid_derivative(layer1)))

	# update the weights 
	weights1 += d_weights1
	weights2 += d_weights2

	
def predict(training_data, weights, threshold=0.5):
    return sigmoid(np.dot(training_data, weights)) >= threshold


X = np.array([[0,0,1],
			  [0,1,1],
			  [1,0,1],
			  [1,1,1]])
y = np.array([[0],[1],[1],[0]])

input      = X
print(input.shape[1])
weights1   = 2*np.random.rand(input.shape[1],4) - 1
print(weights1) 
weights2   = 2*np.random.rand(4,1) - 1               
output     = np.zeros(y.shape)

for i in range(1500):
	layer1,output = feedforward(input,weights1,weights2)
	backprop(input,y,weights1,weights2,layer1,output)

print(output)