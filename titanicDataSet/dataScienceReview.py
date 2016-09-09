import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.svm import SVC
clf = SVC(kernel="linear")
#use rb for python 2.7
csv_file_object = csv.reader(open('train.csv', 'r'))
header = next(csv_file_object)
data=[]

for row in csv_file_object: # Go through each row in the csv file
    data.append(row)        # adding each row to the data variable
data = np.array(data)       # Then convert from a list to an array

# Print all data from column 1 start at row 0
# print(data[0::,1])
# print("----------")
# #Print all data from column 0 and 2
# print(data[:,[0,2]])
# #Print 50 (rows)samples from column 0 and 2
# print(data[:50,[0,2]])
print(data[:,9])
#Need to use a classification algorithm because categorizing
#X is size [n_samples, n_features]
#y of class labels (strings or integers), size [n_samples]
X = data[:,[1,2]]
y = data[:, 0]
# number_passengers = np.size(data[0::,1].astype(np.float))
# number_survived = np.sum(data[0::,1].astype(np.float))
clf.fit(X, y)

# list1 = [1, 2, 3, 4, 5]
# list2 = [1, 2, 3, 4, 5]
# plt.scatter(list1, list2, color='red', marker='o', label='setosa')
# X = X.T #transpose - need to transpose data because of the shape
print(X.shape)

#Graphing
# scatter(x, y, etc.)
# plot_decision_regions(X=X, y=y, clf=clf)
# plt.scatter(data[:, 1], data[:, 2], color='red', marker='o', label='class')
plt.scatter(data[:, 9], data[:, 1], color='blue', marker='o')
plt.ylabel('Survived')
plt.xlabel('Ticket Price')
plt.legend(loc='upper left')
plt.xlim([0,100])
plt.title('SVM in scikit-learn')
plt.show()

#must be executed after graph
plt.xlim([0,500])

#
# pred = clf.predict(features_test)
#
# from sklearn.metrics import accuracy_score
# acc = accuracy_score(pred, labels_test)
