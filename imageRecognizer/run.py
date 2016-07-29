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
Essentially, function f is assigning scores to each class. image is classified well,
if score is high

W(stacked classifiers)

every single score is a weighted sum (counting up colors). matching templates
taking entire row of 3072 and reshape back to image to undo distortion done.
EX. plane turns into blue blob. Counting up blue in image and classifier will show
eventually show +1 -1
INTERESTING - combining all images to create mdoel = blur image . Linear has this problem

if you use linear classifier, use a gray Scale. yellow car might be frog
'''

#linear part 2
#Loss and reg
'''
trying to use scores that will have similar.
minimize - find w that gives lowest loss (measues unhappiness)

Multiclass svm loss (binary support vector class) - this loss score defines how correct
you are. High number is bad, low is good. take the max of all scores - highest score,
then add 1

loss function, just squaring
square loss vs squared hinge loss
number of classes - 1
dont want to inflate score so set to zero
'''

'''
Weight Regularization
measure the niceness of your weight. adding objectives to loss. You append this to your hyperparameter
This will fight with your other weights, but it will improve test performance
l1, l2, elastic net(l1 + l2), max norm regularization
L2 Regularization (weight decay) is most common.
Goal of regularization is to use as many x as possible. Helps to identify image
so [.25,.25.25.25] is perfered over [1,0,0,0], weights diffused
more evidence is being accumulated for evidence
'''
#alter regularization -- internesting


'''
softmax classifier
scores - unormalized log probabilities of the classes
1. get all scores (unormalized log properties) and exponentiate all of them
2. returns unormalized probabilities
3. normalize (divide)
4. returns raw probabilities
5. need to minimize log probability
goal is lower your total loss. but not to zero
'''

'''
Optimization:
Attempting to get to low loss.
Gradient Descent
Compute the slope to go down hill

'''








#ww
