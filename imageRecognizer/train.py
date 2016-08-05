'''
Check list
-Gradient Descent w/ mini batch
    Numerical + analytical gradient (backward pass)
-optimization
-loss function
-backward pass
-step size/learning rate
-back propagation


'''

class Trainer(object):
     """
  A Trainer encapsulates all the logic necessary for training classification
  models. The Trainer performs stochastic gradient descent using different
  update rules defined in optim.py.
  The trainer accepts both training and validataion data and labels so it can
  periodically check classification accuracy on both training and validation
  data to watch out for overfitting.
  To train a model, you will first construct a Trainer instance, passing the
  model, dataset, and various optoins (learning rate, batch size, etc) to the
  constructor. You will then call the train() method to run the optimization
  procedure and train the model.

  After the train() method returns, model.params will contain the parameters
  that performed best on the validation set over the course of training.
  In addition, the instance variable trainer.loss_history will contain a list
  of all losses encountered during training and the instance variables
  solver.train_acc_history and solver.val_acc_history will be lists containing
  the accuracies of the model on the training and validation set at each epoch.

  Example usage might look something like this:

  data = {
    'X_train': # training data
    'y_train': # training labels
    'X_val': # validation data
    'X_train': # validation labels
  }
  model = MyAwesomeModel(hidden_size=100, reg=10)
  trainer = Trainer(model, data,
                  update_rule='sgd',
                  optim_config={
                    'learning_rate': 1e-3,
                  },
                  lr_decay=0.95,
                  num_epochs=10, batch_size=100,
                  print_every=100)
  trainer.train()
  A Trainer works on a model object that must conform to the following API:
  - model.params must be a dictionary mapping string parameter names to numpy
    arrays containing parameter values.
  - model.loss(X, y) must be a function that computes training-time loss and
    gradients, and test-time classification scores, with the following inputs
    and outputs:
    Inputs:
    - X: Array giving a minibatch of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,) giving labels for X where y[i] is the
      label for X[i].
    Returns:
    If y is None, run a test-time forward pass and return:
    - scores: Array of shape (N, C) giving classification scores for X where
      scores[i, c] gives the score of class c for X[i].
    If y is not None, run a training time forward and backward pass and return
    a tuple of:
    - loss: Scalar giving the loss
    - grads: Dictionary with the same keys as self.params mapping parameter
      names to gradients of the loss with respect to those parameters.
  """

  def __init__(self):
      self.model = model
      

  def _reset(self):
      print("reset")

  def _step(self):
      '''
      Make a signle gradient update. Called by train()
      '''
      print("update")

  def train(self):
      '''
      run optimization to train model
      '''
