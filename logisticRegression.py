import numpy as np
from scipy.sparse import csr_matrix, spmatrix, vstack
from math import log
from time import time
from os import path
from collections import OrderedDict
import pickle
from scipy.special import logsumexp
from util import *
import matplotlib.pyplot as plt



# Step0: Set hyper-parameters
LAMBDA = 1.0
NUM_ITER = 10
LEARNING_RATE = 0.00001
NUM_CLASS = 16
np.random.seed(93106) # Can change the seed if you want. This was only for testing purposes.


# Step 1: Preprocess X, Y. Prepare for training.
if path.exists("word_collection.pickle"):
    print("Loading word_collction")
    word_collection = pickle.load(open("word_collection.pickle",'rb'))
else:
    print("Computing word_collction")
    word_collection = generate_word_collection('training.txt')
    pickle.dump(word_collection, open("word_collection.pickle",'wb'))

if path.exists('train_X.pickle'):
    print("Loading train_X and train_Y")
    train_X = pickle.load(open("train_X.pickle", 'rb'))  
    train_Y = pickle.load(open("train_Y.pickle", 'rb')) 

else:
    print("Computing train_X and train_Y")
    train_X, train_Y = data_prepare('training.txt', word_collection)
    pickle.dump(train_X, open("train_X.pickle", 'wb'))
    pickle.dump(train_Y, open("train_Y.pickle", 'wb'))



if path.exists("test_X.pickle"):
    print("Loading test_X and test_Y")
    test_X = pickle.load(open("test_X.pickle", 'rb'))
    test_Y = pickle.load(open("test_Y.pickle", "rb"))
else:
    print("Computing test_X and test_Y")
    test_X, test_Y = data_prepare('testing.txt', word_collection)
    pickle.dump(test_X, open("test_X.pickle", 'wb'))
    pickle.dump(test_Y, open("test_Y.pickle", 'wb'))


train_X = train_X.toarray() # Converting csr_matrices back to np arrays for calculations
test_X = test_X.toarray()
train_X = (train_X - np.mean(train_X))/np.std(train_X) # Normalizing data
test_X = (test_X - np.mean(train_X))/np.std(train_X)

# Step 2: Initialize the vector Theta or load it if the model has already been trained
if path.exists("Theta_SGD.pickle"):
    Theta_SGD = pickle.load(open("Theta_SGD.pickle", 'rb'))
else:
    Theta_SGD = np.zeros([train_X.shape[1], NUM_CLASS])

# Step 3: Train the model with stochastic gradient descent
    train_acc_collection_sgd = []
    test_acc_collection_sgd = []
    time_training_collection_sgd = []
    time_eclipse = 0.0
    N = train_X.shape[0]
    T = time()

    for iter in range(NUM_ITER * N):
        Theta_SGD = train(Theta_SGD, train_X, train_Y, stochastic_gradient, LEARNING_RATE, 1, LAMBDA)
        
        if iter % N == 0:
            time_eclipse += time() - T
            train_acc = evaluation(Theta_SGD, train_X, train_Y)
            test_acc = evaluation(Theta_SGD, test_X, test_Y)

            print(f"Iter {iter} Training data accuracy: {train_acc:0.6f} , Testing data accuracy: {test_acc:0.6f}")

            train_acc_collection_sgd.append(1.0 - train_acc)
            test_acc_collection_sgd.append(1.0 - test_acc)
            time_training_collection_sgd.append(time_eclipse)
            T = time()
    pickle.dump(Theta_SGD, open("Theta_SGD.pickle", 'wb'))


if path.exists("Theta.pickle"):
    Theta = pickle.load(open("Theta.pickle", 'rb'))
else:

# Step 4: Train the model with full vanilla gradient descent. Optional
    Theta = np.zeros([train_X.shape[1], NUM_CLASS])
    train_acc_collection_gd = []
    test_acc_collection_gd = []
    time_training_collection_gd = []
    time_eclipse = 0.0
    N = train_X.shape[0]
    T = time()
    for iter in range(1000):
        Theta = train(Theta, train_X, train_Y, full_gradient, LEARNING_RATE, 1, LAMBDA)
        time_eclipse += time() - T
        train_acc = evaluation(Theta, train_X, train_Y)
        test_acc = evaluation(Theta, test_X, test_Y)

        print(f"Iter {iter} train_acc: {train_acc:0.6f} , Test_error: {test_acc: 0.6f}")

        train_acc_collection_gd.append(1.0 - train_acc)
        test_acc_collection_gd.append(1.0 - test_acc)
        time_training_collection_gd.append(time_eclipse)
        T = time()
    pickle.dump(Theta, open("Theta.pickle", 'wb'))
    
# Step 5: Plot results, check for convergence. This will only run if the graph doesn't exist yet and the model hasn't been trained (no Theta.pickle or Theta_SGD.pickle)
if not path.exists("Error_Over_Time.png"):
    plt.figure()
    plt.plot(time_training_collection_sgd, train_acc_collection_sgd, label='SGD-training error rate')
    plt.plot(time_training_collection_sgd, test_acc_collection_sgd, label='SGD-testing error rate')
    plt.plot(time_training_collection_gd, train_acc_collection_gd, label='GD-training error rate') # Optional
    plt.plot(time_training_collection_gd, test_acc_collection_gd, label='GD-testing error rate') # Optional
    plt.legend()
    plt.xlabel('Training time')
    plt.ylabel('Error rate')
    plt.savefig('Error_Over_Time.png')
    plt.show()

# Step 6: The model is trained! You can now use your new theta values and make inferences. Let's test it out
authorList = {}
with open("authors.txt", 'r') as file:
    for line in file:
        value, key = line.split(',')
        authorList[int(key)] = value

# Choose any quote from one of the 15 authors. Find one online and copy/paste it
userInput = input("Would you like to type your quote here? Type y or n: ")
if userInput == 'y':
    testQuote = input("Put in your quote: ")
else:
    testQuote = "No man, for any considerable period, can wear one face to himself and another to the multitude, without finally getting bewildered as to which may be the true" # You can change this manually if you want
    print("Using default quote in the program: {}".format(testQuote))
test = TF_IDF_encoding(testQuote)
prediction = np.argmax(inference(Theta, test.toarray())) # Could use Theta_SGD as well
print("This quote probably came from {}!".format(authorList[prediction]))
