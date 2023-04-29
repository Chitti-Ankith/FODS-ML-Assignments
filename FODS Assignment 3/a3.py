import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

dfs = pd.read_excel("data.xlsx", sheet_name="Sheet1")
# print(dfs['AP'])
# print(dfs.index) #Use for looping
# print(dfs['RH'][9567])
training_data = np.zeros(shape = (7568,5))
test_data = np.zeros(shape = (2000,5))
t = np.zeros(shape = (7568,1))


def cal_loss(pred,actual):		#calculates mean square error
	sum = 0
	for i in range(2000):
		sum += (actual[i] - pred[i])**2
	loss = sum/2000
	return loss


def iloss(m,test):
	predicted_values = [0 for i in range(2000)]
	for i in range(2000):
		predicted_values[i] = m[0]*test[i][0] + m[1]*test[i][1] + m[2]*test[i][2] + m[3]*test[i][3] + m[4]*test[i][4]
	
	loss = cal_loss(predicted_values,t_test)
	return loss

	
def val_loss(m,training_data,lamb,t,flag = 1):  # For calculating the validation loss
	predicted_values = [0 for i in range(2000)]
	for i in range(2000):
		predicted_values[i] = m[0]*training_data[i][0] + m[1]*training_data[i][1] + m[2]*training_data[i][2] + m[3]*training_data[i][3] + m[4]*training_data[i][4]
		
	if flag == 1:
		loss = cal_loss(predicted_values,t) + lamb*sum(m)
	else:
		for i in range(len(m)):
			m[i] = abs(m[i])
		loss = cal_loss(predicted_values,t) + lamb*sum(m)
	
	return loss

# def g_value(actual,predicted,j):
	
	
def step_gradient_L2(m,training_data,t,learning_rate,lamb): # Step gradient for L2 regularisation
	#gradient descent
	m_gradient = [0 for i in range(5)]
	flag = 0
	for i in range(7568):
		diff = t[i] - (m[0]*training_data[i][0] + m[1]*training_data[i][1] + m[2]*training_data[i][2] + m[3]*training_data[i][3] + m[4]*training_data[i][4])
		m_gradient[0] += -(2/7568)*(diff)
		m_gradient[1] += -(2/7568)*training_data[i][1] * diff
		m_gradient[2] += -(2/7568)*training_data[i][2] * diff
		m_gradient[3] += -(2/7568)*training_data[i][3] * diff
		m_gradient[4] += -(2/7568)*training_data[i][4] * diff
	
	new_m = [0 for i in range(5)]
	# new_m[0] = m[0]
	for i in range(5):		#Updating the coefficient values
		new_m[i] = (1 - 2*lamb*learning_rate)*m[i] - (learning_rate * m_gradient[i]) 
	# if(iloss(new_m,test) - iloss(m,test) < 0.001):
		# flag = 1
	
	return new_m,flag

def step_gradient_L1(m,training_data,t,learning_rate,lamb):	# Step gradient for L1 regularisation
	#gradient descent
	m_gradient = [0 for i in range(5)]
	flag = 0
	for i in range(7568):
		diff = t[i] - (m[0]*training_data[i][0] + m[1]*training_data[i][1] + m[2]*training_data[i][2] + m[3]*training_data[i][3] + m[4]*training_data[i][4])
		m_gradient[0] += -(2/7568)*(diff)
		m_gradient[1] += -(2/7568)*training_data[i][1] * diff
		m_gradient[2] += -(2/7568)*training_data[i][2] * diff
		m_gradient[3] += -(2/7568)*training_data[i][3] * diff
		m_gradient[4] += -(2/7568)*training_data[i][4] * diff
	
	new_m = [0 for i in range(5)]
	# new_m[0] = m[0]
	for i in range(5):	#Updating the coefficient values
		new_m[i] = m[i] - lamb*learning_rate*m[i]/abs(m[i]) - (learning_rate * m_gradient[i]) 
	# if(iloss(new_m,test) - iloss(m,test) < 0.001):
		# flag = 1
	
	return new_m,flag
	
def step_gradient(m,training_data,t,learning_rate,test):	# Step gradient for regular gradient descent
	#gradient descent
	m_gradient = [0 for i in range(5)]

	flag = 0
	for i in range(7568):
		diff = t[i] - (m[0]*training_data[i][0] + m[1]*training_data[i][1] + m[2]*training_data[i][2] + m[3]*training_data[i][3] + m[4]*training_data[i][4])
		m_gradient[0] += -(2/7568)*(diff)
		m_gradient[1] += -(2/7568)*training_data[i][1] * diff
		m_gradient[2] += -(2/7568)*training_data[i][2] * diff
		m_gradient[3] += -(2/7568)*training_data[i][3] * diff
		m_gradient[4] += -(2/7568)*training_data[i][4] * diff
	
	new_m = [0 for i in range(5)]
	for i in range(0,5):	#Updating the coefficient values
		new_m[i] = m[i] - (learning_rate * m_gradient[i]) 
	
	return new_m,flag

def gradient_descent_L1(training_data, m, learning_rate, num_iterations,t,test): 	#Gradient Descent for L1 Regularisation
	
	lamb = [0.0001,0.001,0.005,0.01,0.02,0.05,0.1]	#Possible regularisation coefficients
	val_error = [0 for i in range(7)]	#For storing the validation error
	ml = m #Storing the values of m for each lambda iteration
	final_m = []
	for j in range(len(lamb)):
		print("lambda =",lamb[j])
		m = ml
		while True:
			l1 = val_loss(m,training_data,lamb[j],t)
			m,f = step_gradient_L1(m,training_data,t,learning_rate,lamb[j])
			
			l = val_loss(m,training_data,lamb[j],t)
			# print(l)
			if(abs(l - l1) < 0.01):		#Stop the process if the loss dosent change by much
				print("Validation Loss:",l)
				print("Coefficients:",m)
				val_error[j] = l
				final_m.append(m)
				break
				
	return val_error,final_m,lamb
	
	
	
def gradient_descent(training_data, m, learning_rate, num_iterations,t,test,flag = False):
	if flag == True :  #L2 Regularisation Descent
		lamb = [0.001,0.005,0.01,0.02,0.05,0.1] #Possible regularisation coefficients
		val_error = [0 for i in range(6)]	#For storing the validation error
		m1 = m	#Storing the values of m for each lambda iteration
		final_m = []
		# print(len(lamb))
		for j in range(len(lamb)):
			print("lambda =",lamb[j])
			m = m1
			while True:
				l1 = val_loss(m,training_data,lamb[j],t)
				m,f = step_gradient_L2(m,training_data,t,learning_rate,lamb[j])
				# print(m)
				
				l = val_loss(m,training_data,lamb[j],t)
				# print(l)
				if(abs(l - l1) < 0.01): #Stop the process if the loss dosent change by much
					print("Validation Loss:",l)
					print("Coefficients:",m)
					val_error[j] = l
					final_m.append(m)
					break
		
		return val_error,final_m,lamb
	else:	#Normal Gradient Descent
		for i in range(num_iterations):	# Repeat the descent process for a given number of iterations
			m,f = step_gradient(m,training_data,t,learning_rate,test)
			# print(m)
			# if f == 1:     #Checking if gradient dosent change by much
				# break
		return m


for i in range(7568):		#Reading the training data
	training_data[i][0] = 1
	training_data[i][1] = dfs['AT'][i]
	training_data[i][2] = dfs['V'][i]
	training_data[i][3] = dfs['AP'][i]
	training_data[i][4] = dfs['RH'][i]
	t[i][0] = dfs['PE'][i]

it = 0
t_test = []
for i in range(7568,9568):	#Reading the Test data
	test_data[it][0] = 1
	test_data[it][1] = dfs['AT'][i]
	test_data[it][2] = dfs['V'][i]
	test_data[it][3] = dfs['AP'][i]
	test_data[it][4] = dfs['RH'][i]
	t_test.append(dfs['PE'][i])
	it += 1
	
#####################################################################Regression###########################################################################	
w1 = np.matmul(np.transpose(training_data),t)
w2 = np.matmul(np.transpose(training_data),training_data)
w2 = np.linalg.inv(w2)
w = np.matmul(w2,w1)   #Contains the Ml estimates of the coefficients

predicted_test_values = [0 for i in range(2000)]
for i in range(2000):
	predicted_test_values[i] = w[0]*test_data[i][0] + w[1]*test_data[i][1] + w[2]*test_data[i][2] + w[3]*test_data[i][3] + w[4]*test_data[i][4]

loss = cal_loss(predicted_test_values,t_test)
print("Part A Loss:",loss)

########################################################Gradient Descent###############################################################################
t_l = []

print("Started Gradient Descent")

for i in np.array(t).flat:  #Converting the matrix to a list so that future functions dont have any problems
	t_l.append(i)
	

cm = np.mean(training_data,axis = 0) #Column wise mean for feature selection
csd = np.std(training_data,axis = 0)  #Column wise standard deviation
cmt = np.mean(test_data,axis = 0) 
csdt = np.std(test_data,axis = 0)
	
for i in range(7568):			#Feature Selection
	training_data[i][0] = 1
	training_data[i][1] = (dfs['AT'][i] - cm[1])
	training_data[i][2] = (dfs['V'][i] - cm[2])
	training_data[i][3] = (dfs['AP'][i] - cm[3])
	training_data[i][4] = (dfs['RH'][i]	- cm[4])

	

m = [0 for i in range(5)]

learning_rate = 0.003
num_iterations = 500
m = gradient_descent(training_data, m, learning_rate, num_iterations,t_l,test_data,False)
m[0] = w[0]
# m[0] = 454.44
print(m)
for i in range(2000):
	predicted_test_values[i] = m[0]*test_data[i][0] + m[1]*test_data[i][1] + m[2]*test_data[i][2] + m[3]*test_data[i][3] + m[4]*test_data[i][4]
	
loss = cal_loss(predicted_test_values,t_test)
print("Part B Loss:",loss)


######################################################L2 Regularisation ########################################################################

m = [0 for i in range(5)]
learning_rate = 0.1
num_iterations = 100

print("Started L2 Regularisation")

for i in range(7568):			#Feature Selection
	training_data[i][0] = 1
	training_data[i][1] = (dfs['AT'][i] - cm[1])/csd[1]
	training_data[i][2] = (dfs['V'][i] - cm[2])/csd[2]
	training_data[i][3] = (dfs['AP'][i] - cm[3])/csd[3]
	training_data[i][4] = (dfs['RH'][i]	- cm[4])/csd[4]


for i in range(2000):			#Feature Selection for test data
	test_data[i][0] = 1
	test_data[i][1] = (test_data[i][1] - cmt[1])/csdt[1]
	test_data[i][2] = (test_data[i][2] - cmt[2])/csdt[2]
	test_data[i][3] = (test_data[i][3] - cmt[3])/csdt[3]
	test_data[i][4] = (test_data[i][4] - cmt[4])/csdt[4]

val_e,m,lamb = gradient_descent(training_data, m, learning_rate, num_iterations,t_l,test_data,True)
# print(m)


print("Regularisation Coefficient :",lamb[np.argmin(val_e)])
# print(lamb)
# print(val_e)

for j in range(1):	
	loss = val_loss(m[j],test_data,lamb[j],t_test)
	print("Loss for L2 Regularisation",loss)

	
plt.plot(lamb,val_e)
plt.xlabel('Regularisation Coefficient') 

plt.ylabel('Loss') 
plt.title('L2 Regularisation')
plt.show() 
#######################################################L1 Regularisation ######################################################################

m = [1 for i in range(5)]
num_iterations = 100
print("Started L1 Regularisation")

learning_rate = 0.1
val_e,m,lamb = gradient_descent_L1(training_data, m, learning_rate, num_iterations,t_l,test_data)
# print(m)

print("Regularisation Coefficient :",lamb[np.argmin(val_e)])
for j in range(1):	
	loss = val_loss(m[j],test_data,lamb[j],t_test)
	print("Loss for L1 Regularisation",loss)
	
plt.plot(lamb,val_e)
plt.xlabel('Regularisation Coefficient') 

plt.ylabel('Loss') 
plt.title('L1 Regularisation')
plt.show() 


	