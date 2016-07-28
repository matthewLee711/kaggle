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
l1 - manhatten
l2 - euclidian
NN gets most similar and do majority votes on it
knn tend to have good performance test time

take train data and try out a lot of hyperparameters

separate traing data into folds and test it on validation data.
don't touch test data because of generalization
cross validation - used when have littel train data

Don't use kNN on pictures with high pixel count.
-if distance metrics are high on objects
-very inefficient
'''

'''
-Image Classification: We are given a Training Set of labeled images, asked
to predict labels on Test Set. Common to report the Accuracy of predictions
(fraction of correctly predicted images)
(people don't use cross validation all the time, they use a single validation)

- We introduced the k-Nearest Neighbor Classifier, which predicts the labels
based on nearest images in the training set

- We saw that the choice of distance and the value of k are hyperparameters
that are tuned using a validation set, or through cross-validation if the size
of the data is small.

- Once the best set of hyperparameters is chosen, the classifier is evaluated
once on the test set, and reported as the performance of kNN on that data.
'''
#need to build CNN and RNN
#Linear Classification
#Parametric approach - take image and produce weigh
#f(x,W) = Wx - takes in image and returns 10 class scores
#f(3072,10)
'''
Taking image and streching pixels into one column of pixel vector 3072 numbers
f(x, W) = 10, 1 --- 3072 numbers going into w (3072 x 1)
Wx = 10 x 3072
+b (bias - independent weight, bias for cat centered data would choose over dog)

W(stacked classifiers)

every single score is a weighted sum (counting up colors). matching templates
taking entire row of 3072 and reshape back to image to undo distortion done.
EX. plane turns into blue blob. Counting up blue in image and classifier will show
eventually show +1 -1
INTERESTING - combining all images to create mdoel = blur image . Linear has this problem

if you use linear classifier, use a gray Scale. yellow car might be frog
'''










#ww
