import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from scipy import stats
# -------Fishers Linear Discriminant----------------#
def FLDA(d_no):
   plt.figure()
   print("For Dataset " + str(d_no))
   dataset = "dataset_"+str(d_no) +".csv" 
   df1 = pd.read_csv(dataset,header = None) #Reading Data
   l11=[]
   l12=[]
   l21=[]
   l22=[]
   t1=[]
   t2=[]
   for i in range(len(df1)):
      if(df1[3][i]==0):
         l11.append(df1[1][i])
         l12.append(df1[2][i])
         t1.append(df1[3][i])
      else:
         l21.append(df1[1][i])
         l22.append(df1[2][i])
         t2.append(df1[3][i])
   # Seperating the classes
   class1=[l11,l12]
   class2=[l21,l22]
   # Plotting the points
   plt.scatter(l11,l12,marker='s',s=100,c='grey',edgecolor='black')
   plt.scatter(l21,l22,marker='^',s=100,c='yellow',edgecolor='black')
   #Finding the mean for class1
   x_mean=0
   y_mean=0
   z=len(class1[0])
   for i in range(z):
      x_mean+=class1[0][i]
      y_mean+=class1[1][i]
   x_mean=x_mean/z
   y_mean=y_mean/z
   class1_mean=np.array([x_mean,y_mean]).reshape(2,1)
   #Finding the mean for class2
   x_mean=0
   y_mean=0
   z=len(class2[0])
   for i in range(z):
      x_mean+=class2[0][i]
      y_mean+=class2[1][i]
   x_mean=x_mean/z
   y_mean=y_mean/z
   class2_mean=np.array([x_mean,y_mean]).reshape(2,1)

   #plotting the 2 means
   plt.scatter(class1_mean[0],class1_mean[1],marker='s',c='black',edgecolor='black')
   plt.scatter(class2_mean[0],class2_mean[1],marker='^',c='black',edgecolor='black')

   #calculating the Individual variances and summing them up
   class1_variance=np.zeros(shape=(2,2))
   for i in range(len(class1[0])):
      a= class1[0][i]
      b= class1[1][i]
      a = a-class1_mean[0][0]
      b = b-class1_mean[1][0]
      class1_variance[0][0] = class1_variance[0][0] + a*a
      class1_variance[0][1] = class1_variance[0][1] + a*b
      class1_variance[1][0] = class1_variance[1][0] + b*a
      class1_variance[1][1] = class1_variance[1][1] + b*b

   class2_variance=np.zeros(shape=(2,2))
   for i in range(len(class2[0])):
      a= class2[0][i]
      b= class2[1][i]
      a = a-class2_mean[0][0]
      b = b-class2_mean[1][0]
      class2_variance[0][0] = class2_variance[0][0] + a*a
      class2_variance[0][1] = class2_variance[0][1] + a*b
      class2_variance[1][0] = class2_variance[1][0] + b*a
      class2_variance[1][1] = class2_variance[1][1] + b*b

   #Finding the Sw
   Sw = class1_variance + class2_variance
   # Finding the inverse of Sw
   Sw_inv = np.linalg.inv(Sw)
   
   #Finding the w vector values

   w = np.matmul(Sw_inv,class2_mean-class1_mean)
   W = np.squeeze(np.matmul(Sw_inv,class2_mean-class1_mean)) #here this W (capital w) is used in projecting the points
   w= w/math.sqrt(w[0][0]**2 + w[1][0]**2) #Normalising the w
   print("w:")
   print(w)
   # calculating the values of y i.e. w transpose . x
   # these y values are used in calculating pdfs
   y_class1=[]
   for i in range(len(class1[0])):
      x = np.array([class1[0][i],class1[1][i]]).reshape(2,1)
      y_class1.append(np.matmul(w.T,x))  
   y_class2=[]
   for i in range(len(class2[0])):
      x = np.array([class2[0][i],class2[1][i]]).reshape(2,1)
      y_class2.append(np.matmul(w.T,x))
   #plotting the projects of points
   for i in range(len(class1[0])):
      x1 = class1[0][i]
      x2 = class1[1][i]
      point = [x1,x2]
      proj = np.dot(point,W)/np.dot(W,W) * W
      y = np.dot(point,W)
      plt.scatter(proj[0],proj[1],color='grey',alpha=0.15)

   for i in range(len(class2[0])):
      x1 = class2[0][i]
      x2 = class2[1][i]
      point = [x1,x2]
      proj = np.dot(point,W)/np.dot(W,W) * W
      y = np.dot(point,W)
      plt.scatter(proj[0],proj[1],color='yellow',alpha=0.15)

   #Converting the y values into normal list
   y1=[]
   for i in range(len(y_class1)):
      y1.append(y_class1[i][0][0])
   y2=[]
   for i in range(len(y_class2)):
      y2.append(y_class2[i][0][0])

   y1 = np.sort(y1)
   y2 = np.sort(y2)
   mean1 = np.mean(y1)
   mean2 = np.mean(y2)
   std1 = np.std(y1)
   std2 = np.std(y2)
   filename = "points for dataset_" + str(d_no)+".png"
   plt.savefig(filename)
   plt.figure() #new figure
   const1=[]
   const2=[]
   for i in range(len(y1)):
      const1.append(1)
   for i in range(len(y2)):
      const2.append(1)
   #plotting the projections separately
   plt.scatter(y1,const1,color='grey')
   plt.scatter(y2,const2,color='yellow')
   filename = "projections for dataset_" + str(d_no)+".png"
   plt.savefig(filename)
   plt.figure()

   #Finding and plotting the PDF's
   pdf = stats.norm.pdf(y1,mean1,std1)
   plt.plot(y1,pdf,color='grey')
   pdf = stats.norm.pdf(y2,mean2,std2)
   plt.plot(y2,pdf,color='yellow')

   # Finding the point of intersetion of 2 normal curves
   a = 1/(2*std1**2) - 1/(2*std2**2)
   b = mean2/(std2**2) - mean1/(std1**2)
   c = mean1**2 /(2*std1**2) - mean2**2 / (2*std2**2) - np.log(std2/std1)
   threshold = np.roots([a,b,c])

   if(threshold[0]>mean1 and threshold[0]<mean2):
      threshold=threshold[0]
   elif (threshold[0] < mean1 and threshold[0]> mean2):
      threshold=threshold[0]
   else:
      threshold=threshold[1]
   #plotting the threshold
   plt.plot(threshold,stats.norm.pdf(threshold,mean1,std1),'o',color = 'black')
   print("threshold")
   print(threshold)
   filename = "Normal Curves for dataset_" + str(d_no)+".png"
   plt.savefig(filename)
   return 1
  
#----------------Main Function----------------#
for i in range(3):
   FLDA(i+1)
