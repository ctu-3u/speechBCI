import numpy as np
import pandas as pd
import random

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


"""Linear transform for alignment"""
## Affine transformations
def similarity_metric(X_transformed, X_target):
    # X_transformed and X_target are all trials in the same condition. Not averaged across trials.
    X_transformed = np.mean(X_transformed, axis = 0)
    X_target = np.mean(X_target, axis = 0)
    return np.mean((X_transformed - X_target) ** 2)

def linear_transform(X, params):    
    # Applies an affine transformation to X using given parameters.
    # Parameters:
    # - X: (N, d) array of source data
    # - params: Flattened array containing A (dxd) and b (dx1)
    # Returns:
    # - Transformed (N, d) array
    d = X.shape[1]
    A = params[:d**2].reshape(d, d)  # Reshape first d^2 elements into a (dxd) matrix
    b = params[d**2:].reshape(1, d)  # Last d elements form (1xd) translation vector
    return X @ A + b # Apply affine transformation

def optimize_transformation(X_source, X_target, label_source, label_target, lamb = 0.001):
    # Finds the optimal affine transformation maximizing a given similarity metric.
    # Parameters:
    # - X_source: Original dataset (N, d)
    # - X_target: Target dataset (N, d)
    # - label1: phoneme labels for arrays in X_source
    # - label2: phoneme labels for arrays in X_target
    # Returns:
    # - Optimal affine transformation matrix A and translation vector b
    d = X_source.shape[1] # TODO: what if X_source and X_target have different size 0?

    conds = np.unique(label_source)

    X_tild = np.ones((len(conds), d + 1))
    Y_tild = np.ones((len(conds), d))

    for i in conds:
        xi = X_source[label_source[()] == i] # all trials under condition i in session 2
        yi = X_target[label_target[()] == i] # all trials under condition i in session 1
        X_tild[i, :-1] = np.mean(xi, axis = 0)
        Y_tild[i, :] = np.mean(yi, axis = 0)


    TransMat = np.linalg.solve(X_tild.T @ X_tild + lamb * np.eye(d+1), X_tild.T @ Y_tild)

    A_opt = TransMat[:-1, :]
    b_opt = TransMat[-1, :]

    return A_opt, b_opt

def transform_alignsessions(X_source, X_target, label_source, label_target, lamb = 0.001):
    # do linear transform to X_source onto X_targer for alignment
    A_opt, b_opt = optimize_transformation(X_source, X_target, label_source, label_target, lamb=lamb)

    X_transformed = X_source @ A_opt + b_opt # 800 trials, 128 channels

    return X_transformed, A_opt, b_opt


"""Proportional Sampling"""
def proportion_sample(datai, label, fraction):
    data = [datai[i] for i in range(datai.shape[0])]
    df = pd.DataFrame({'data': data, 'label': label})
    sampled_df = df.groupby('label', group_keys=False).apply(\
        lambda x: x.sample(frac=fraction, random_state=16))

    sampled_data = np.vstack(sampled_df['data'].values)
    sampled_label = sampled_df['label'].values

    return sampled_data, sampled_label


"""PCA"""
def pca_2sessions(feature1, feature2, num_pcs = 39):
    if num_pcs != feature1.shape[1]:
        # Fit PCA to the 2 sessions respectively
        pcas1 = PCA(n_components=num_pcs)
        pcas1.fit(feature1)
        pcas2 = PCA(n_components=num_pcs)
        pcas2.fit(feature2)

        # Transform the arrays separately into PCA space
        feature1_pca = pcas1.transform(feature1)  # shape will be (640, 39)
        feature2_pca = pcas2.transform(feature2)  # shape will be (800, 39)

        return feature1_pca, feature2_pca
    else:
        return feature1, feature2
    

"""Train classifier and test"""
def classification_train_test(feature1, label1, feature2, label2, trainset1, trainset2, testset2, \
                              bool_trans = False, folds = 10, clasf = RandomForestClassifier, **kwargs):
    # trainset1: number of samples for training from session 1. so as trainser2 and testset2.
    # kwargs: lamb
    num_sess1 = feature1.shape[0]
    num_sess2 = feature2.shape[0]

    if trainset2 + testset2 > num_sess2 or trainset1 > num_sess1:
        raise Exception("Training set and test set size should not sum to exceeding total number of samples.")
    if testset2 < len(np.unique(label2)):
        raise Exception("Training set and test set size shouldnt be less that number of classification groups.")
    if testset2 == 0:
        raise Exception("Test set size shouldnt be 0.")
    if trainset1 == 0 and trainset2 == 0: # no training. just give chance.
        return random.choices(range(1, len(np.unique(label2)+1)), k=int(testset2*folds)), \
                             np.ones(int(testset2*folds))
    trainsize_1 = trainset1 / num_sess1
    trainsize_2 = trainset2 / num_sess2
    testsize = testset2 / num_sess2
    
    y_test_log = []
    y_pred_log = []
    for _ in range(folds):
        # extract training set and test set
        if trainset1 == 0:
            X_train_1, y_train_1 = None, None
        elif trainset1 != 0 and trainset1 < len(np.unique(label1)):
            raise Exception("Traing set and test set size shouldn't be less that number of classification groups.")
        else:
            X_train_1, _, y_train_1, _ = train_test_split(feature1, label1, \
                                                            train_size=trainsize_1, stratify = label1)
            
        if trainset2 == 0:
            X_train_2, y_train_2 = None, None
            _, X_test, _, y_test = train_test_split(feature2, label2, \
                                                            test_size=testsize, stratify = label2)
        elif trainset2 != 0 and trainset2 < len(np.unique(label2)):
            raise Exception("Traing set and test set size shouldn't be less that number of classification groups.")
        else:
            X_train_2, X_test, y_train_2, y_test = train_test_split(feature2, label2, \
                                                            train_size=trainsize_2, test_size=testsize, stratify = label2)
        
        # train classifier
        clf = clasf()
        if X_train_1 is None:
            X_train, y_train = X_train_2, y_train_2
            if bool_trans:
                raise Exception("Provide session 1 data for transformation targeting.")
        elif X_train_2 is None:
            X_train, y_train = X_train_1, y_train_1
        else:
            y_train = np.append(y_train_1, y_train_2)
            if not bool_trans:
                X_train = np.concatenate((X_train_1, X_train_2), axis = 0)
            else:
                lamb = kwargs.get('lamb', 0.001)
                X_train_2, A_opt, b_opt = transform_alignsessions(\
                    X_train_2, X_train_1, y_train_2, y_train_1, lamb=lamb)
                X_train = np.concatenate((X_train_1, X_train_2), axis = 0)
                X_test = X_test @ A_opt + b_opt

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_test_log = np.append(y_test_log, y_test)
        y_pred_log = np.append(y_pred_log, y_pred)

    return y_test_log, y_pred_log


"""Examine direct-sampling classifiers with different training set generation"""
def examine_classifiers(feature1, label1, feature2, label2, filename, \
                        trainsets_1, trainsets_2, testsets_2, bools_trans, rounds=10, **kwargs):
    # trainsets_1: list of training set size from session 1. numpy array.
    # trainsets_2: list of training set size from session 2. numpy array.
    # testsets_2: list of test set size from session 2. numpy array.
    # bools_trans: list of booleans indicating whether to do linear transformation.
    # these 4 lists should have the same length.
    # kwargs: folds, lamb
    if len(set([len(trainsets_1), len(trainsets_2), len(testsets_2), len(bools_trans)])) != 1:
        raise Exception("Make sure input lists of training set size, test set size and tranform booleans have the same length.")
    num_trials = len(trainsets_1)

    accuracylists = np.zeros((num_trials, rounds + 2))
    tabletitle = []

    for i in range(num_trials):
        tabletitle.append(f"Tr:1-{trainsets_1[i]}+2-{trainsets_2[i]}, Te:2-{testsets_2[i]}. Trans:{bools_trans[i]}")
        for j in range(rounds):
            folds = kwargs.get('folds', 10)
            lamb = kwargs.get('lamb', 0.001)
            y_test, y_pred = classification_train_test(feature1, label1, feature2, label2, \
                                                       trainsets_1[i], trainsets_2[i], testsets_2[i], bools_trans[i], folds, lamb=lamb)
            accuracylists[i, j] = accuracy_score(y_test, y_pred)
        accuracylists[i, -2] = np.mean(accuracylists[i, :-2]) # mean value of all rounds
        accuracylists[i, -1] = np.std(accuracylists[i, :-2]) # stdev of all rounds

    
    dataframe = {}
    for i in range(num_trials):
        dataframe[tabletitle[i]] = accuracylists[i,:]
    df = pd.DataFrame(dataframe)
    df.to_excel(filename, index = False)

    return accuracylists






'''TODO'''
"""Examine sub-sampling classifiers with different training set generation"""
def subclassifier(X_train, X_test, y_train, y_test, *args): # args are in the order of: sub-level 0, sub-level 1, ...
    # train
    clf0 = RandomForestClassifier()
    X0_train = X_train[np.isin(y_train, args[0])]
    y0_train = y_train[np.isin(y_train, args[0])]
    clf0.fit(X0_train, y0_train)

    clf1 = [RandomForestClassifier() for _ in range(len(args[0]))]
    for i in range(len(clf1)):
        X1_train = X_train[np.isin(y_train, args[1][i])]
        y1_train = y_train[np.isin(y_train, args[1][i])]
        clf1[i].fit(X1_train, y1_train)

    # test
    y0_pred = clf0.predict(X_test)
    y1_pred = []
    for i in range(len(X_test)):
        y1_pred.append(clf1[args[0].index(y0_pred[i])].predict(X_test[i].reshape(1, -1)))

    return np.array(y1_pred)
    

















def examine_subclassifier(features1, labels1, features2, labels2, filename, *args): # args are in the order of: classifier function, sub-level 0, sub-level 1, ...
    accuracylists = np.zeros((5,5))
    for i in range(5):
        ### take training samples from session1 and session 2
        X_train, _, y_train, _ = train_test_split(features1, labels1, \
                                                        test_size = 0.2, stratify = labels1)

        y_test_log = []
        y_pred_log = []
        for _ in range(10):
            X_test, y_test = proportion_sample(features2, labels2, 0.16)
            
            clf = RandomForestClassifier()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            y_test_log = np.append(y_test_log, y_test)
            y_pred_log = np.append(y_pred_log, y_pred)

        accuracy = accuracy_score(y_test_log, y_pred_log)
        accuracylists[i, 0] = accuracy
        trainsize = 0
        for j in range(4):
            trainsize += 0.2
            testsize = 1 - trainsize

            ### take training samples from session1 and session 2
            X_train_1, _, y_train_1, _ = train_test_split(features1, labels1, \
                                                        test_size = 0.2, stratify = labels1)
            X_train_2, X_test21, y_train_2, y_test21 = train_test_split(features2, \
                                                                        labels2, test_size = testsize, stratify = labels2)

            X_train = np.concatenate((X_train_1, X_train_2), axis = 0)
            y_train = np.append(y_train_1, y_train_2)

            y_test_log = []
            y_pred_log = []
            for _ in range(10):
                X_test, y_test = proportion_sample(X_test21, y_test21, 0.16 / testsize)
                
                y_pred = args[0](X_train, X_test, y_train, y_test, *args[1:])
                
                y_test_log = np.append(y_test_log, y_test)
                y_pred_log = np.append(y_pred_log, y_pred)


            accuracy = accuracy_score(y_test_log, y_pred_log)
            accuracylists[i, j+1] = accuracy
            

    tabletitle = ["Chance", "Tr160", "Tr320", "Tr480", "Tr640"]

    dataframe = {}
    for i in range(5):
        dataframe[tabletitle[i]] = accuracylists[:, i]
    df = pd.DataFrame(dataframe)
    df.to_excel(filename, index = False)

