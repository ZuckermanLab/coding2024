import numpy as np
from itertools import product

def generate_data(n_sample:int=1000, n_features:int=3, rand_seed:int=123, threshold:float=1.5):
    """
    Helper function to generate a sample data set.

    Returns: 
        X: an array shape n_samples by n_features
        y: a 1-D array length n_samples
    """
    # generate random synthetic data for training
    np.random.seed(rand_seed)
    X = np.random.rand(n_sample, n_features)

    # make two classes based on the sum of each feature in the data
    # if the features for one observation sum to > 1.5, then assign the class label "1". Otherwise the class label is "0"
    y = (X.sum(axis=1) > 1.5).astype(int) 
    return(X, y)

def sigmoid_activation(z):
  return 1/(1 + np.exp(-z))

def predict(X, weights):
  z = np.dot(X, weights)
  return (sigmoid_activation(z) > 0.5).astype(int)

def accuracy_score(true_labels, pred_labels):
  return sum(true_labels == pred_labels)/len(true_labels)

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
    # define the bounds of our grid search for each parameter
    weight_range = np.linspace(weights_lwr, weights_upr, weights_nsteps)

    # obtain the weights and accuracy of our model by a grid search
    best_weights, best_accuracy = grid_search(train_data, train_labels, weight_range)

    # let's see how we did!
    print(f"Best weights: {best_weights}")
    print(f"Best accuracy: {best_accuracy}")
    return(best_weights, best_accuracy)

def test_neuron(X_test, y_test, best_weights):
    test_predictions = predict(X_test, best_weights)
    prediction_accuracy = accuracy_score(y_test, test_predictions)
    print(f"Accuracy on the test data: {100*np.round(prediction_accuracy, 2)}%")
    return(prediction_accuracy)

# main
def main():
   X_train, y_train = generate_data()
   weights, accuracy = train_neuron(X_train, y_train)
   X_test, y_test = generate_data(rand_seed=321)
   prediction_accuracy = test_neuron(X_test=X_test, y_test=y_test, best_weights=weights)

main()
