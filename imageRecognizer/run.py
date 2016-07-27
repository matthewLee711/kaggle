'''
SIFT - Scale invariant feature transform
Features are detected and and represented in patches
as numerical vectors. Sift descripters are used to convert
each patch into a 128-dimensional vector.
'''
#old school^

#new method: Data driven
'''
Training a model
kNN - manhatten (l1) distance.
compare all pixel values and subtract from each other
compare value differences
X = images
y = label
hyperparameter: l1 or l2 distance

NN gets most similar and do majority votes on it
knn tend to have good performance test time

take train data and try out a lot of hyperparameters
'''
