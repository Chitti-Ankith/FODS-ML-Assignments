"""
This is the solution to Assignment 4 FODS. Part A plots the distribution of the data by sending them all at once.
Part B plots the distribution of the data by passing the data sequentially.
"""

# Importing the required libraries
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
import scipy.special as ssp

# Defining the beta function
def beta(m,l):
    ap = []
    for mew in xax:
        ap.append((ssp.gamma(m+a+l+b)/(ssp.gamma(m+a)*ssp.gamma(l+b)))* mew**(m+a-1)*(1-mew)**(l+b-1))
    return ap

# c1 is the number of heads and c2 is the number of tails
c1 = 0
ls = []
# Total number of points
n = 150

# Generating a random 150 points
for i in range(n):
    ls.append(np.random.randint(0,2))
    if(ls[i]==1):
        c1+=1
c2 = n-c1

# Taking a and b that satisfy the given conditions (mean = 0.4)
a = 4
b = 6

#############################################################
## Sending all the values at once (PART A)
#Defining the x axis for plotting
xax = [mew for mew in np.arange(0,1,0.0001)]

out = beta(c1,c2)
# Plotting
# print(out)
# plt.scatter(xax,out)
# plt.show()

#############################################################
# Reseting the values of number of heads and number of tails
count0 = 0
count1 = 0
## Sequentially sending the data (PART B)
for i in ls:
    if(i==1): count1+=1
    else: count0+=1
    o = beta(count0,count1)
    # Plot each of the graphs in a directory called images
    # plt.scatter(xax,o)
    # plt.savefig("./images/{}.png".format(count0+count1+1000))
