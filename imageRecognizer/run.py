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
Optimization: 3-50
Attempting to get to low loss.
Numerical gradient
Compute the slope to go down hill
Cons:
    approximate
    slow
Numerical gradient:
    approximate
    slow
Analytic gradient:
    exact, fast, error prone
ALWAYS USE analytic gradient, but use numerical gradient to check it (gradient check)
run both at same time and compare AND MAKE sure they are the same (pass gradient check)

To develop a new module for neural network:
    1. get loss
    2. get backward pass that computes gradient
    3. Then do gradient check to make sure the calculus was correct

Gradient Descent

step_size/learning rate AND weight regularization strength are difficult to manage
and what we cross validate over
step size is much step to take when moving in gradient direction

Mini batch gradient Descent
-only use small portion of training set to compute gradient -- noisy though, but more steps
-this tends to work better.
Use specific minibatch size to fit on your gpu.
The issue which exists is finding the correct learning rate
    -Too high learning rate willresult in a lot of loss
    -Really low learning rate will result in good loss, but its only ok
    -so do a fast learning rate, then switch to low learning (parameter update/momentum)
    keep track of velocity (adagrad seems good)

Now we can do linear classifiers...
'''

'''
History - HOG/SIFT find edges then create histograms based on those
'''


#LECTURE 3 BACKPROPAGATION
#conv nets
'''
Important
currently:
score function
loss function
data loss + regularization function
optimization: gradient Descent
     gradient is a fancy word for derivative, or the rate of change of a function
     https://betterexplained.com/articles/vector-calculus-understanding-the-gradient/

Computational graphs
-forward pass -left to right recursive

Backpropagation
Computing gradient of expressions through recursive application of chain rule

Vectorized Operations: Jacobian Matrix
Communincation between the gates are vectors, but they only care about gradient
In practice we process entire minibatch of 100 examples at one time so 100 4096d output vectors
409600x409600
'''

'''
Assignment writing svm/softmax 49:35

neural nets will be very large: no hope of writing down gradient formula by
hand for all parameters
- backpropagation = recursive application of the chain rule along a
computational graph to compute the gradients of all
inputs/parameters/intermediates
- implementations maintain a graph structure, where the nodes implement
the forward() / backward() API.
- forward: compute result of an operation and save any intermediates
needed for gradient computation in memory
- backward: apply the chain rule to compute the gradient of the loss
function with respect to the inputs.
'''

'''
Neural network - scoring

Linear score function:
f = Wx
(more complex matethmatical expression of x)
2 layer neural network:
f = W_2 max(0,W_1(x))

Recieve input x, multiplied by matrix
do matrix multiply and threshold of everything negative to zero is activation function,
and then do one more matrix multiply
This gives us our score

Matrix multiply gives us 100 numbers and another matrix multiply to get scores
The more layers you have, the more matrix multiplies you do

With linear scoring, you only have one number for comparision.
Ex. red car forward backwards, side, etc.
With neural networks, you have hundreds of numbers and each can be for a different car.
2 layers
elements of h (hyperparamter become positive to more similarities found)
h is also size of hidden parameter(we get to choose also want as big as possible)
W1
W2 sum accross all car templates

neurons turn on and off if they find a specific score

BY Adding more layers, you can compute more interesting function from the image

'''

'''
Assignment: Writing 2 layer net
train with backpropagation
*generate weights: you need to use

ReLu is faster than sigmoid. Default non linearilty
count layers by which ones have weights
kernel trick, changing your data so it is linearly separable


we arrange neurons into fully-connected layers
- the abstraction of a layer has the nice property that it
allows us to use efficient vectorized code (e.g. matrix
multiplies)
- neural networks are not really neural
- neural networks: bigger = better (but might have to
regularize more strongly)
'''


'''
'''



#ww
