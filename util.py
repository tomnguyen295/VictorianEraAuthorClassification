import numpy as np
from scipy.sparse import csr_matrix, spmatrix, vstack
from math import log
from time import time
from os import path
from collections import OrderedDict
import pickle
from scipy.special import logsumexp
import matplotlib.pyplot as plt

IDF = dict()

# First, we have to model our data into something that we can use for matrix operations. We will use TF-IDF encoding for this
def generate_word_collection(file_name):
    '''
    @input:
        file_name: a string. should be either "training.txt" or "testing.txt"
    @return:
        word_collection: use the data structure you find proper to repesent the words 
        as well as how many times the word appears in a given file.
    '''
    fhand = open(file_name, encoding='ISO-8859-1')
    wordCounts = dict()
    for line in fhand:
        lineParse = line.split()
        lineParse.pop(-1)
        for word in lineParse:
            wordCounts[word] = wordCounts.get(word, 0) + 1
    
    index = 0
    for word in wordCounts:
        wordCounts[word] = index
        index += 1

    return wordCounts

word_collection = generate_word_collection('training.txt') # We use this to keep track of the total count of all words in our vocabulary. Needed for TF-IDF

def bag_of_word_feature(sentence):
    """
    @input:
        sentence: a string. Stands for "D" in the problem description. 
                            One example is "wish for solitude he was twenty years of age ".
    @output:
        encoded_array: a sparse vector based on library scipy.sparse.csr_matrix.
    """
    global word_collection
    data = np.zeros(0)
    indices = np.zeros(0)
    word_split = sentence.strip().split()

    tempDict = dict()

    for word in word_split:
        if word in word_collection:
            tempDict[word] = tempDict.get(word, 0) + 1   
    
    for key,value in tempDict.items():
        indices = np.append(indices, list(word_collection).index(key))
        data = np.append(data, [value])
    
    indptr = np.array([0, len(data)])

    encoded_array = csr_matrix((data, indices, indptr),(1,len(word_collection)))
    encoded_array.sort_indices()
    return encoded_array
    


def get_TF(term, document=None):
    """
    @input:
        term: str or list of str. words (e.g., [cat, dog, fish, are, happy])
        document: list of str. a sentence (e.g., ["wish", "for", "solitude").
            None if identical to term
    @output:
        TF: some datastructure containing frequency of each term in the document.
    """
    if isinstance(term, str):
        term = [term]
    if document is None:
        document = term
    if (not type(document) is list):
        document = document.strip().split()

    totalTerms = len(document)
    termCounts = dict()
    TF = dict()

    for word in term:
        if word in word_collection:
            termCounts[word] = termCounts.get(word, 0) + document.count(word) 
            if word not in TF:
                TF[word] = termCounts[word]/totalTerms 
    return TF

def get_IDF(file_name):
    """
    @input:
        file_name: a string. should be either "training.txt" or "texting.txt"
    @output:
        None. Update the global variable you defined
    """
    global IDF
    num_document = 0

    with open(file_name, encoding='ISO-8859-1') as f:
        for line in f:
            num_document += 1
            line = line.strip().split(',')[0] # list of words in the document
            line = set(line.split())
            for word in line:
                IDF[word] = IDF.get(word,0) + 1
        for key in IDF:
            IDF[key] = np.log10(num_document/IDF[key])



def get_TF_IDF(term):
    """
    @input:
        term: str or list of str. words (e.g., [cat, dog, fish, are, happy])
    @output:
        TF_IDF: csr_matrix. Equal to TF*IDF.
    """
    if isinstance(term, str):
        term = [term]
    if (not type(term) is list):
        term = term.strip().split()

    data = np.zeros(0)
    indices = np.zeros(0)
    global IDF
    for tf in get_TF(term):
        try:
            data = np.append(data, [get_TF(term)[tf]*IDF[tf]])
            indices = np.append(indices, word_collection[tf])
        except KeyError:
            pass # if a word is not in the vocabulary, ignore it.
    indptr = np.array([0, len(data)])
    encoded_array = csr_matrix((data,indices,indptr), (1,len(word_collection)))
    encoded_array.sort_indices()
    return encoded_array


def TF_IDF_encoding(document):
    """
    @input:
        document: str. a sentence (e.g., "wish for solitude he was twenty years of age ").
        file_name: a string. should be either "training.txt" or "texting.txt"
    @output:
        encoded_array: a sparse vector based on library scipy.sparse.csr_matrix. 
        Contain the TF_IDF_encoding of the given document.
    """
    global IDF    
    return get_TF_IDF(document.split())

if path.exists("IDF.pickle"):
    IDF = pickle.load(open("IDF.pickle", 'rb'))
else:  
    get_IDF('training.txt')  
    pickle.dump(IDF, open("IDF.pickle",'wb')) # We create an IDF dictionary for all words in our vocabulary



def data_prepare(file_name, word_collection):
    """
    @input:
        file_name: a string. should be either "training.txt" or "texting.txt"
        word_collection: a list. Refer to the output of generate_word_collection(file_name).
    @return:
        X: a numpy array. size of (N,d)
        Y: a numpy array. size of (N,k)
    """
    X = []
    Y = []
    with open(file_name, encoding='ISO-8859-1') as f:
        for line in f:
            input_string, y = line.strip().split(',')
            x = TF_IDF_encoding(input_string)
            X.append(x)
            Y.append(int(y))
    X = vstack(X)
    Y = np.array(Y)
    return X, Y

def get_error_rate(pred, label):
    """
    @input:
        pred: 1-D np.ndarray with dtype int. Predicted labels.
        label: 1-D np.ndarray with dtype int. The same shape as pred. Ground truth labels.
    @return:
        error rate: float. 
    """
    resultVector = np.abs(pred - label)
    errors = np.sum(resultVector)
    return float(errors) / len(pred)  

def log_softmax(x):
    """
    @input:
        x: a 2-dimension numpy array
    @output:
        result: a numpy array of the same shape as x
    Compute log softmax of x along second dimension.
    Using this function provides better numerical stability than using softmax
    when computing cross entropy loss
    """
    
    return x - logsumexp(x, axis=1, keepdims=True)

def get_one_hot(Y, k):
    if not isinstance(Y, np.ndarray) or len(Y.shape) == 0:
        Y = np.array([Y])
    b = Y.shape[0]
    return csr_matrix((np.ones(b), (np.arange(b), Y)), shape=[b, k])

def inference(theta, x):
    """
    @input:
        theta: a numpy matrix of shape [d,k]
        x: a numpy.ndarray of shape [d] or [b, d]
            if you're not comfortable handing x as 2-d array, 
            you can assume x is a numpy.array of shape [d]
    @output:
        result: a numpy array of shape [b, k].
        d = # of features
        k = # of classifications
        b = # of training examples
    """
    # delete me if x is always 1-d
    if len(x.shape) == 1:
        x = x.reshape(1, -1) # convert to shape [b, d]
    
    hypothesis = np.dot(x, theta)

    return log_softmax(hypothesis)

def gradient(x, y, Theta):
    """
    gradient of cross-entropy loss with respect to Theta
    @input:
        x: a numpy array. size of (d,) or (b, d)
        y: a numpy array. size of (k,) or (b, k)
        Theta: a numpy array. size of (d,k)
        if you're not comfortable handing x and y as 2-d array, 
            you can assume x and y is a numpy.array of shape [d]
    @output:
        grad: gradient w.r.t. parameter Theta. size of (d,k)
        d = # of features
        k = # of classifications
    """
    
    # delete me if x is always 1-d
    if len(x.shape) == 1:
        x = x.reshape(1, -1) # convert to shape [b, d]
    if len(y.shape) == 1:
        y = y.reshape(1, -1) # convert to shape [b, d]

    if (np.shape(x)[1] != 10000):
        x = np.transpose(x)

    if (np.shape(Theta)[1] != 16):
        Theta = np.transpose(Theta)

    preError = np.exp(inference(Theta, x)) - y
    grad = np.dot(np.transpose(x), preError)

    return grad

def full_gradient(Theta, X, Y, Lambda):
    """
    @input:
        Theta: a numpy array. size of (d,k)
        X: a numpy array. size of (b,d)
        Y: a numpy array. size of (b,k)
    @return:
        gradient_sum: a numpy array. size of (d,k). Full gradient to Theta, averaged over all data points. 
    """
    k = Theta.shape[1]
    Y_one_hot = get_one_hot(Y, k)

    grad = gradient(X, Y_one_hot, Theta) + 2*Lambda*Theta
    return grad

def stochastic_gradient(Theta, X, Y, Lambda):
    """
    @input:
        Theta: a numpy array. size of (d,k)
        X: a numpy array. size of (N,d)
        Y: a numpy array. size of (N,k)
    @return:
        gradient_sum: a numpy array. size of (d,k). Stochastic gradient to Theta on a single sampled data point.
    """

    k = Theta.shape[1]
    b = X.shape[0]
    idx = np.random.randint(b)
    y = get_one_hot(Y[idx], k)
    x = np.reshape(X[idx, :], (-1,1))
    grad = b * (gradient(x, y, Theta)) + 2*Lambda*Theta

    return grad

def train(Theta, X, Y, gradient_function, learning_rate, num_iter, Lambda):
    """


    @input:
        Theta: a numpy array. size of (d,k)
        X: a numpy array. size of (N,d)
        Y: a numpy array. size of (N,)
        gradient_function:  a function. Should be either "full_gradient", "stochastic_gradient"
        learning_rate: a float.
        num_iter: an integer. Number of iterations.
        Lambda: a float. The regularization term.
    @return: 
        Update_Theta: a numpy array. size of (d,k). Updated parameters.
    """
    for _ in range(num_iter):
        gradient_update = gradient_function(Theta, X, Y, Lambda)
        Theta = Theta - (learning_rate * gradient_update)
    return Theta  

def evaluation(Theta, X, Y):
    """
    @input:
        Theta: a numpy array. size of (d,k)
        X: a numpy array. size of (d,d)
        Y: a numpy array. size of (d,)
    @output:
        eval_acc: a float. evaluation accuracy.
    """


    Y = np.reshape(Y,(-1,1))
    Y_hat = inference(Theta, X)
    Y_argmax = np.argmax(Y_hat, axis=1)
    eval_acc = float(np.sum(Y_argmax == Y))/len(Y)
    return eval_acc