import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def pi_c(n):  #Calulates Pi for Circle
	x=[]
	y=[]
	for i in range(n):   #Randomly generates numbers between -2 and 2
		x.append(np.random.uniform(-2,2))
		y.append(np.random.uniform(-2,2))
	a=0
	r=2					#r is the radius value which has been set to 2
	col = []
	
	for i in range(n):     
		if x[i]**2 + y[i]**2 <= r**2 :
			#col = np.where(x[i]**2 + y[i]**2 <= r**2,'b','r')
			col.append('b')  			#Updating the color to blue if the points lie within the circle radius
			a=a+1
		else:
			col.append('r')
			
		i=i+1
	a=a*4
	i=a/n
	plt.scatter(x,y,c=col)   #Shows the Plot
	plt.show()
	return i


def pi_s(n):  #Calculates Pi value for Sphere
	x=[]
	y=[]
	z=[]
	for i in range(n):    
		x.append(np.random.uniform(-2,2))
		z.append(np.random.uniform(-2,2))
		y.append(np.random.uniform(-2,2))
	a=0
	r=2
	col = []
	ax = plt.axes(projection='3d')

	for i in range(n):
		if x[i]**2 + y[i]**2 + z[i]**2 <= r**2 :
			a=a+1
			col.append('b')			#Updating the color to blue if the points lie within the sphere radius
		else:
			col.append('r')
		i=i+1
	a=a*6
	i=a/n
	ax.scatter3D(x,y,z,c=col)   #For plotting the 3d plot
	plt.show()
	return i

value = 10
for i in range(7):  #Considering samples from 10 to 10^7
	print("Sample Size :" + str(value**(i+1)))
	res1 = pi_c(value**(i+1))
	res2 = pi_s(value**(i+1))
	print("Circle estimate: " + str(res1) + " \nSphere Estimate: " + str(res2))



