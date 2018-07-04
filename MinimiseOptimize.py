# -*- coding: utf-8 -*-
'''

'''

import numpy as np
import sklearn.metrics as metrics
from scipy.optimize import minimize
from sklearn.metrics import log_loss


def MinimiseOptimize(preds,models_filenames,nb_classes,sample_N,testY,NUM_TESTS = 20):
    '''
    preds ,此时为cifar_100 (6,10000, 100)
    testY,类别列表，100分类，就是：[[1],[2],[3],...],(100,1)
    nb_classesf,分类数,100
    models_filenames模型名称,5
    NUM_TESTS , 迭代次数，默认为20次
    '''
    best_acc = 0.0
    best_weights = None
    # Parameters for optimization
    constraints = ({'type': 'eq', 'fun':lambda w: 1 - sum(w)})
    bounds = [(0, 1)] * len(preds)

    # Check for NUM_TESTS times
    for iteration in range(NUM_TESTS):  # NUM_TESTS,迭代次数为25
        # Random initialization of weights
        prediction_weights = np.random.random(len(models_filenames))

        # Minimise the loss 
        result = minimize(log_loss_func, prediction_weights, method='SLSQP', bounds=bounds, constraints=constraints)

        weights = result['x']
        weighted_predictions = np.zeros((sample_N, nb_classes), dtype='float32')

        # Calculate weighted predictions
        for weight, prediction in zip(weights, preds):
            weighted_predictions += weight * prediction

        yPred = np.argmax(weighted_predictions, axis=1)
        yTrue = testY

        # Calculate weight prediction accuracy
        accuracy = metrics.accuracy_score(yTrue, yPred) * 100
        recall = recall_score(yTrue, yPred)

        print('\n ------- Iteration : %d  - acc: %s  - rec:%s -------  '%((iteration + 1),accuracy,recall))
        print('    Best Ensemble Weights: \n',result['x'])
        
        # Save current best weights 
        if accuracy > best_acc:
            best_acc = accuracy
            best_weights = weights
    return best_acc,best_weights

# Create the loss metric 
def log_loss_func(weights):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = np.zeros((sample_N, nb_classes), dtype='float32')

    for weight, prediction in zip(weights, preds):
        final_prediction += weight * prediction

    return log_loss(testY_cat, final_prediction)

# calculate accuracy/recall
def calculate_weighted_accuracy(prediction_weights):
    '''计算acc/recall 以及得到预测结果'''
    weighted_predictions = np.zeros((sample_N, nb_classes), dtype='float32')
    for weight, prediction in zip(prediction_weights, preds):
        weighted_predictions += weight * prediction
    yPred = np.argmax(weighted_predictions, axis=1)
    yTrue = testY
    accuracy = metrics.accuracy_score(yTrue, yPred) * 100
    recall = recall_score(yTrue, yPred)
    print("Accuracy : ", accuracy)
    print("Recall : ", recall)
    

