import numpy as np
from itertools import product

def generate_data(n_sample:int=1000, n_features:int=3, rand_seed:int=123, threshold:float=1.5):
    """
    Helper function to generate a sample data set.

    Returns: 
        X: an array shape n_samples by n_features
        y: a 1-D array length n_samples
    """
    pass
    # generate random synthetic data for training
    # set the random seed!
    
    
    # generate an array n_samples by n_features with random numbers
    # X = 

    # make two classes based on the sum of each feature in the data
    # if the features for one observation sum to > threshold, then assign the class label "1". Otherwise the class label is "0"
    # y =  
    #return X, y

def sigmoid_activation(x):
  pass
  # calculate the sigmoid of x
  # sig_x = 
  # return sig_x 

def predict(X, weights):
    pass
  # calculate the dot product of X and the weights
  # dot_product = 

  # Round to zero or 1 if the sigmoid is >0.5
  # rounded = 

  # return rounded.astype(int)

def accuracy_score(true_labels, pred_labels):
  pass
  # calculate the number of true labels that matched the predicted labels
  # n_correct = 

  # divide by the number of samples
  # perc_correct = 
  
  #return perc_correct

def grid_search(X, y, weight_range):
  best_weights = None
  best_accuracy = 0

  for w1, w2, w3 in product(weight_range, repeat=3):
    weights = np.array([w1, w2, w3])
    predictions = predict(X, weights)
    accuracy = accuracy_score(y, predictions)
    if accuracy > best_accuracy:
      best_accuracy = accuracy
      best_weights = weights
  return best_weights, best_accuracy

def train_neuron(train_data, train_labels, weights_lwr:int= -10, weights_upr:int=10, weights_nsteps:int = 100):
    pass
    # define the bounds of our grid search for each parameter
    # weight_range = 

    # obtain the weights and accuracy of our model by a grid search
    #best_weights, best_accuracy = 

    # let's see how we did!
    #print(f"Best weights: {best_weights}")
    #print(f"Best accuracy: {best_accuracy}")
    #return(best_weights, best_accuracy)

def test_neuron(X_test, y_test, best_weights):
    pass
    # test_predictions = 
    # prediction_accuracy = 
    # print(f"Accuracy on the test data: {100*np.round(prediction_accuracy, 2)}%")
    # return(prediction_accuracy)

# main
def main():
   X_train, y_train = generate_data()
   weights, accuracy = train_neuron(X_train, y_train)
   X_test, y_test = generate_data(rand_seed=321)
   prediction_accuracy = test_neuron(X_test=X_test, y_test=y_test, best_weights=weights)

main()
