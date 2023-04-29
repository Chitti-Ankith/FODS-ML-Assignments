# Importing the necessary libraries
import pandas as pd
import numpy as np
import sys
from heapq import heappush, heappop
from operator import itemgetter
import math

pd.set_option('display.expand_frame_repr', False)

# Reading the data and cleaning the data using pandas
# We estimate the missing values using the average of all the values in that coloumn.

df = pd.read_csv('madrid_2016.csv')
df['BEN'].fillna((df['BEN'].mean()), inplace=True)
df['CO'].fillna((df['CO'].mean()), inplace=True)
df['EBE'].fillna((df['EBE'].mean()), inplace=True)
df['NMHC'].fillna((df['NMHC'].mean()), inplace=True)
df['NO'].fillna((df['NO'].mean()), inplace=True)
df['NO_2'].fillna((df['NO_2'].mean()), inplace=True)
df['O_3'].fillna((df['O_3'].mean()), inplace=True)
df['PM10'].fillna((df['PM10'].mean()), inplace=True)
df['PM25'].fillna((df['PM25'].mean()), inplace=True)
df['SO_2'].fillna((df['SO_2'].mean()), inplace=True)
df['TCH'].fillna((df['TCH'].mean()), inplace=True)
df['TOL'].fillna((df['TOL'].mean()), inplace=True)

# This is the normalisation of the dataframe. In the case of normalisation we subtract the mean and divide the variance in the required coloumns.
# df = pd.read_csv('madrid_2016.csv')
# df['BEN'].fillna((df['BEN'].mean()), inplace=True)
# df['BEN'] = (df['BEN']-df['BEN'].mean())/df['BEN'].std()
# df['CO'].fillna((df['CO'].mean()), inplace=True)
# df['CO'] = (df['CO']-df['CO'].mean())/df['CO'].std()
# df['EBE'].fillna((df['EBE'].mean()), inplace=True)
# df['EBE'] = (df['EBE']-df['EBE'].mean())/df['EBE'].std()
# df['NMHC'].fillna((df['NMHC'].mean()), inplace=True)
# df['NMHC'] = (df['NMHC']-df['NMHC'].mean())/df['NMHC'].std()
# df['NO'].fillna((df['NO'].mean()), inplace=True)
# df['NO'] = (df['NO']-df['NO'].mean())/df['NO'].std()
# df['NO_2'].fillna((df['NO_2'].mean()), inplace=True)
# df['NO_2'] = (df['NO_2']-df['NO_2'].mean())/df['NO_2'].std()
# df['O_3'].fillna((df['O_3'].mean()), inplace=True)
# df['O_3'] = (df['O_3']-df['O_3'].mean())/df['O_3'].std()
# df['PM10'].fillna((df['PM10'].mean()), inplace=True)
# df['PM10'] = (df['PM10']-df['PM10'].mean())/df['PM10'].std()
# df['PM25'].fillna((df['PM25'].mean()), inplace=True)
# df['PM25'] = (df['PM25']-df['PM25'].mean())/df['PM25'].std()
# df['SO_2'].fillna((df['SO_2'].mean()), inplace=True)
# df['SO_2'] = (df['SO_2']-df['SO_2'].mean())/df['SO_2'].std()
# df['TCH'].fillna((df['TCH'].mean()), inplace=True)
# df['TCH'] = (df['TCH']-df['TCH'].mean())/df['TCH'].std()
# df['TOL'].fillna((df['TOL'].mean()), inplace=True)
# df['TOL'] = (df['TOL']-df['TOL'].mean())/df['TOL'].std()

# print(df[:3])
stations = list(df['station'])
df = df.drop(['station'],axis =1)
# This is used to convert stations from a categorical variable into a quantitative variable.
df2 = pd.get_dummies(stations)
df = pd.concat([df, df2], axis=1)
df.sort_values(by=['date'], inplace=True, ascending=True)
df = df[:-24]

# Reading the data and cleaning the data using pandas
# We estimate the missing values using the average of all the values in that coloumn.

df2 = pd.read_csv('madrid_2017.csv')
df2.sort_values(by=['date'], inplace=True, ascending=True)
df2['BEN'].fillna((df2['BEN'].mean()), inplace=True)
df2['CO'].fillna((df2['CO'].mean()), inplace=True)
df2['EBE'].fillna((df2['EBE'].mean()), inplace=True)
df2['NMHC'].fillna((df2['NMHC'].mean()), inplace=True)
df2['NO'].fillna((df2['NO'].mean()), inplace=True)
df2['NO_2'].fillna((df2['NO_2'].mean()), inplace=True)
df2['O_3'].fillna((df2['O_3'].mean()), inplace=True)
df2['PM10'].fillna((df2['PM10'].mean()), inplace=True)
df2['PM25'].fillna((df2['PM25'].mean()), inplace=True)
df2['SO_2'].fillna((df2['SO_2'].mean()), inplace=True)
df2['TCH'].fillna((df2['TCH'].mean()), inplace=True)
df2['TOL'].fillna((df2['TOL'].mean()), inplace=True)
df2 = df2[:552]
stations = list(df2['station'])
df3 = pd.get_dummies(stations)
df4 = df2.drop(['station'],axis =1)
df3.reset_index(drop=True, inplace=True)
df4.reset_index(drop=True, inplace=True)
df3 = pd.concat([df4, df3], axis=1)
df4 = df4.drop(['PM25'],axis =1)
df4 = df4.drop(['CH4'],axis =1)
df4 = df4.drop(['date'],axis =1)
df4 = df4.drop(['NOx'],axis =1)

# This is the normalisation of the dataframe. In the case of normalisation we subtract the mean and divide the variance in the required coloumns.
# df2 = pd.read_csv('madrid_2017.csv')
# df2.sort_values(by=['date'], inplace=True, ascending=True)
# df2['BEN'].fillna((df2['BEN'].mean()), inplace=True)
# df2['BEN'] = (df2['BEN']-df2['BEN'].mean())/df2['BEN'].std()
# # df2['CH4'].fillna((df2['CH4'].mean()), inplace=True)
# df2['CO'].fillna((df2['CO'].mean()), inplace=True)
# df2['CO'] = (df2['CO']-df2['CO'].mean())/df2['CO'].std()
# df2['EBE'].fillna((df2['EBE'].mean()), inplace=True)
# df2['EBE'] = (df2['EBE']-df2['EBE'].mean())/df2['EBE'].std()
# df2['NMHC'].fillna((df2['NMHC'].mean()), inplace=True)
# df2['NMHC'] = (df2['NMHC']-df2['NMHC'].mean())/df2['NMHC'].std()
# df2['NO'].fillna((df2['NO'].mean()), inplace=True)
# df2['NO'] = (df2['NO']-df2['NO'].mean())/df2['NO'].std()
# df2['NO_2'].fillna((df2['NO_2'].mean()), inplace=True)
# df2['NO_2'] = (df2['NO_2']-df2['NO_2'].mean())/df2['NO_2'].std()
# df2['O_3'].fillna((df2['O_3'].mean()), inplace=True)
# df2['O_3'] = (df2['O_3']-df2['O_3'].mean())/df2['O_3'].std()
# df2['PM10'].fillna((df2['PM10'].mean()), inplace=True)
# df2['PM25'].fillna((df2['PM25'].mean()), inplace=True)
# df2['PM25'] = (df2['PM25']-df2['PM25'].mean())/df2['PM25'].std()
# df2['SO_2'].fillna((df2['SO_2'].mean()), inplace=True)
# df2['SO_2'] = (df2['SO_2']-df2['SO_2'].mean())/df2['SO_2'].std()
# df2['TCH'].fillna((df2['TCH'].mean()), inplace=True)
# df2['TCH'] = (df2['TCH']-df2['TCH'].mean())/df2['TCH'].std()
# df2['TOL'].fillna((df2['TOL'].mean()), inplace=True)
# df2['TOL'] = (df2['TOL']-df2['TOL'].mean())/df2['TOL'].std()

# We can obtain the closest k neighbours using a heap and popping out the best k values.

def predict(k,h):
    avg = 0
    for j in range(k):
        x = heappop(h)
        y=x[1]
        print("y is ",y)
        # print(type(df3['PM25'][0]))
        z = df3['PM25'][y]
        # p = z[1]
        # avg+=df3['PM25'][heappop(h)[1]]
        # avg+=df3['PM25'][y]
        avg+=z
    avg /= k
    print(avg)
    return avg

def predict2(df3,kval):
    ans2 = []
    temp = []
    expected = []
    # the total number of estimations to be made on the 2107 data
    estimations = 552

    for i in range(estimations):
        df5 = list(df4.iloc[i])
        # This function calculates the euclidean distance on all the points in the dataframe with respect to a single point in the 2017 data.
        a = np.linalg.norm(df3[['BEN', 'CO', 'EBE', 'NMHC', 'NO','NO_2','O_3','PM25', 'SO_2', 'TCH', 'TOL']].sub(np.array(df5)), axis=1)
        z=[]
        for j in range(len(a)):
            z.append((a[j],j))
        p = sorted(z,key=lambda x: x[0])
        h = ()
        x=[]
        for k in range(kval):
            h = p[k]
            x.append(h[1])
        y=0
        for t in range(kval):
            if(x[t] < 552 ):
                test = df3.iloc[x[t]]
                y = y+ test['PM25']
                pass
            else:
                test = df2.iloc[x[t]-552]
                y = y+ test['PM25']
        expected.append(y/kval)
        temp.append(i)
        df3 = df3.append(df4.iloc[i-1])
    # To find the best value of k we calculate the RMS value between several values of K and take the best one
    error = 0
    orig = list(df2['PM25'][:])
    # print(len(orig))
    # print(len(expected))
    for i in range(len(orig)):
        error += (orig[i]-expected[i])**2
    print(math.sqrt(error/estimations))
    # print(expected)

series = [5,7,9,10,12,15,20]
for i in series:
    predict2(df3,i)
