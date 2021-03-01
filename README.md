# VictorianEraAuthorClassification
Using NLP and Multiclass Logistic Regression to classify quotes belonging to 15 Victorian-era authors

Every algorithm involved in this project was implemented from scratch (no sklearn, tensorflow, or any other machine learning/deep learning library)

Originally a project for UCSB's AI class CS165A, modified for the purposes of this repo and for usability

# Description 
The goal of this model was to learn how to classify different quotes and texts from 15 different authors and correctly decide which author it belongs to. The training data used to train this model consists of 3000 text instances containing roughly 1000 words each ending with a comma and a label from 1 to 15 representing each of the authors. The cross validation data used to test this model contains 500 text instances of the same format.

# The Process
All of the text data was converted into matrices using TF-IDF encoding so mathematical operations could be performed on the data. To do this, we need to create our vocabulary using a dictionary that contains every unique word found in our training text data. We have our feature space be the TF-IDF values of each word in our dictionary. After the encoding, the model can be trained using either vanilla gradient descent or stochastic gradient descent for learning the parameters attributed to each of the features. Inferences used for this model utilize the softmax activation function since this is most optimal for multiclass regression. Similarly, the cost function used for this was the cross-entropy loss function. Both regular gradient descent and stochastic gradient descent can then be used to minimize this loss function and find the optimal weight vector for our model. This weight vector has been saved via "pickling" and can be used to make future predictions.

# Results
Here's what happened to the training and cross-validation error over time as the weight vector kept updating. Pretty solid.

![plot](https://github.com/tomnguyen295/VictorianEraAuthorClassification/blob/main/Error_Over_Time.png)


# Usage
The model has already been trained! To use as is, just run linearRegression.py, find a quote online from one of the 15 authors (authors can be found in authors.txt) and copy/paste it in when prompted to. Now of course the model won't be right 100% of the time, but it actually does a decent job. If you want to see the entire training process yourself though, go ahead and delete Theta.pickle, Theta_SGD.pickle, and Error_Over_Time.png. You can mess around with any of the hyperparameters too if you are curious. Now, if you want to see the encoding process.... well good luck because this part took my computer a couple hours to run for some reason. Just delete the rest of the .pickle files for this (HIGHLY RECOMMEND THAT YOU DON'T BUT UP TO YOU)
