import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from sklearn.svm import SVC
clf = SVC(kernel="linear")

#df = pd.read_csv("train.csv", skipinitialspace=True)

#use rb for python 2.7
csv_file_object = csv.reader(open('train.csv', 'r'))
header = next(csv_file_object)
data=[]

for row in csv_file_object: # Go through each row in the csv file
    data.append(row)        # adding each row to the data variable
data = np.array(data)       # Then convert from a list to an array

#print(data[:,9])
#Need to use a classification algorithm because categorizing
print(data[:,[1,2]])
#X is size [n_samples, n_features]
#y of class labels (strings or integers), size [n_samples]
X = data[:,[1,2]]
y = data[:, 0]
clf.fit(X, y)

#Create mesh to plot in
h = .02  # step size in the mesh
x_min = X[:,0].min()
x_max = X[:,0].max()
# y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
# xx, yy = np.meshgrid(np.arrange(x_min, x_max, h),
#                     np.arrange(y_min, y_max, h))

#Graphing data points
# plot_decision_regions(X=X, y=y, clf=clf)
# plt.scatter(data[:, 1], data[:, 2], color='red', marker='o', label='class')
plt.scatter(data[:, 9], data[:, 1], color='blue', marker='o')
plt.ylabel('Survived')
plt.xlabel('Ticket Price')
plt.legend(loc='upper left')
plt.xlim([0,100])
plt.title('SVM in scikit-learn')
plt.show()


#Calculate accuracy
# pred = clf.predict(features_test)
#
# from sklearn.metrics import accuracy_score
# acc = accuracy_score(pred, labels_test)
