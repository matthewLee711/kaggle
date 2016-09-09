import pandas as pd
import numpy as np
import pylab as P

#Always use header=0 when you know row 0 is the header row
df = pd.read_csv('train.csv', header=0)

#Acquire first 10 rows f Age column
#df['Age'][0:10]
#print df.info()

# Number of males in each class
for i in range(1,4):
    print i, len(df[(df['Sex'] == 'male') & (df['Pclass'] == i) ])

#display graph
# df['Age'].hist()
# P.show()

#Drop missing values of Age
# df['Age'].dropna().hist(bins=16, range=(0,80), alpha=.5)
# P.show()

df['Gender'] = df['Sex'].map( lambda x : x[0].upper() )
df['Gender'] = df['Sex'].map( {'female' : 0, 'male': 1} ).astype(int)

#Calculate median of ages
median_ages = np.zeros((2,3))
"""
array([[ 0., 0., 0.],
       [ 0., 0., 0.]])
"""
for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = df[(df['Gender'] == i) & \
                              (df['Pclass'] == j+1)]['Age'].dropna().median()
"""
array([[ 35. , 28. , 21.5],
       [ 40. , 30. , 25. ]])
"""
#Make copy of age
df['AgeFill'] = df['Age']
df.head()

df[ df['Age'].isnull() ][['Gender','Pclass','Age','AgeFill']].head(10)
