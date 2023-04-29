import pandas as pd
import numpy as np

LEARNING_RATE=0.000001
STOP=15



def read_dataset():
	data=pd.read_excel("Folds5x2_pp.xlsx",sheet_name="Sheet1")
	train_x=np.zeros((7560,5))
	train_t=np.zeros((7560,1))
	test_x=np.zeros((2000,5))
	test_t=np.zeros((2000,1))
	#w=np.zeros((5,1))
	for row in data.itertuples():
		#	(row)
		if row[0]<7560:
			train_x[row[0]][0]=1
			for i in range(4):
				train_x[row[0]][i+1]=row[i+1]
			train_t[row[0]]=row[5]
		elif row[0]<9560:
			test_x[row[0]-7560][0]=1
			for i in range(4):
				test_x[row[0]-7560][i+1]=row[i+1]
			test_t[row[0]-7560]=row[5]

	#print(train_t)

	return train_x,train_t,test_x,test_t




def normal_eq(X,T):
	A=np.matmul(X.T,X)
	A=np.linalg.inv(A)
	B=np.matmul(X.T,T)
	W=np.matmul(A,B)
	return W


def err(X,W,T):
	Y=np.matmul(X,W)
	#print(Y)		
	E=np.subtract(T,Y)
	E=np.matmul(E.T,E)
	print(E[0][0]/np.shape(Y)[0])
	return E[0][0]/np.shape(Y)[0]
	err=0
	for row in E:
		err=err+(row[0]**2)
	return err/np.shape(Y)[0]	


def grad_desc(X,T):
	W=np.random.rand(5,1)
	prev_err=err(X,W,T)
	curr_err=prev_err-1
	while prev_err-curr_err>0.00001:
		prev_err=curr_err
		Y=np.matmul(X,W)
		A=np.subtract(Y,T)
		delta=np.matmul(X.T,A)
		W=np.subtract(W,(LEARNING_RATE/np.shape(Y)[0])*delta)
		curr_err=err(X,W,T)
	return W



def L1W(X,T,lamda):
	W=np.random.rand(5,1)	
	prev_err=err(X,W,T)
	curr_err=prev_err-1
	while prev_err-curr_err>0.00001:
		prev_err=curr_err
		Y=np.matmul(X,W)
		A=np.subtract(Y,T)
		delta=np.matmul(X.T,A)


def L1grad_desc(X,T):
	X1=X[:7000,:]
	T1=T[:7000,:]
	T2=T[7000:,:]
	X2=X[7000:,:]
	x=[i for i in np.arange(0,30,1)]
	y=[]
	W_MAX=np.zeros((5,1))
	min_err=100000
	for lamda in x:
		W=L1W(X1,T1,lamda)
		e=err(X2,W,T2)
		y.append(e)
		if min_err>e:
			W_MAX=W
			min_err=e
	return W_max,x,y
	#print(np.shape(X2))

X,T,test_X,test_T=read_dataset()
W=normal_eq(X,T)
print(err(test_X,W,test_T))
#W1=grad_desc(X,T)
print("YOOO")
L1grad_desc(X,T)
#print(err(test_X,W1,test_T))
print(err(test_X,W,test_T))	