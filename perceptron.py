#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #4
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:,-1]  #getting the last field to form the class label for training

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the data by using Pandas library

X_test = np.array(df.values)[:,:64]    #getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]     #getting the last field to form the class label for test


singleLayer = [True, False] #is single layer boolean values

for i in n: #iterates over n

    for j in r: #iterates over r

        #iterates over both algorithms
        #-->add your Pyhton code here

        for k in singleLayer : #iterates over the algorithms if values are true or false

            #Create a Neural Network classifier
            #if Perceptron then

            #   clf = Perceptron()    #use those hyperparameters: eta0 = learning rate, shuffle = shuffle the training data, max_iter=1000
            #else:
            #   clf = MLPClassifier() #use those hyperparameters: activation='logistic', learning_rate_init = learning rate, hidden_layer_sizes = number of neurons in the ith hidden layer,
            #                          shuffle = shuffle the training data, max_iter=1000
            #-->add your Pyhton code here
            if singleLayer:
                clf = Perceptron(eta0=i, shuffle=j, max_iter=1000)
            else:
                clf = MLPClassifier(max_iter=1000, activation="logistic", learning_rate_init=i)

            #Fit the Neural Network to the training data
            clf.fit(X_training, y_training)

            #make the classifier prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously with zip() Example:
            #for (x_testSample, y_testSample) in zip(X_test, y_test):
            #to make a prediction do: clf.predict([x_testSample])
            #--> add your Python code here
            count = 0 #correct counts
            for (x_testSample, y_testSample) in zip(X_test, y_test):
                count += clf.predict([x_testSample]) == y_testSample

            #check if the calculated accuracy is higher than the previously one calculated for each classifier. If so, update the highest accuracy
            #and print it together with the network hyperparameters
            #Example: "Highest Perceptron accuracy so far: 0.88, Parameters: learning rate=0.01, shuffle=True"
            #Example: "Highest MLP accuracy so far: 0.90, Parameters: learning rate=0.02, shuffle=False"
            #--> add your Python code here
            scb = -1 #single correct best
            MLPcb = -1 #multi layered perceptron correct best
            if scb< count and singleLayer:
                print("Highest Perceptron accuracy so far: ", (count / len(y_test)), ", Parameters: learning rate=", i, ", shuffle=", j, sep='')
                singleCorrectBest = count
            if (MLPcb< count and not singleLayer):
                print("Highest MLP accuracy so far: ", (count / len(y_test)), ", Parameters: learning rate=", i, ", shuffle=", j, sep='')
                MLPcb = count











