import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

import localtool.operator as loper

"""t_SNE visualiztaion"""
def transformation_tsne(X_source, X_target, A_opt, b_opt, data_session1, data_session2, perplexity=30, n_iter=1000):
    # Apply the optimized transformation
    X_transformed = X_source @ A_opt.T + b_opt

    # Concatenate all datasets for t-SNE transformation
    combined_data = np.vstack([X_source, X_target, X_transformed])
    
    # Perform t-SNE to reduce dimensionality to 2D
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    combined_data_2d = tsne.fit_transform(combined_data)
    
    # Split the transformed data
    X_source_2d = combined_data_2d[:len(X_source)]
    X_target_2d = combined_data_2d[len(X_source):len(X_source) + len(X_target)]
    X_transformed_2d = combined_data_2d[len(X_source) + len(X_target):]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(X_source_2d[:, 0], X_source_2d[:, 1], label="Original Source Data", alpha=0.6, \
                c=data_session2['trialCues'], cmap='turbo', marker='s', linewidths=0)
    plt.scatter(X_target_2d[:, 0], X_target_2d[:, 1], label="Target Data", alpha=0.6, \
                c=data_session1['trialCues'], cmap='turbo', marker='o', linewidths=0)
    plt.scatter(X_transformed_2d[:, 0], X_transformed_2d[:, 1], label="Transformed Source Data", alpha=0.6, \
                c=data_session2['trialCues'], cmap='turbo', marker='x')
    
    plt.title("Comparison of Original, Target, and Transformed Data (t-SNE Reduced)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend()
    plt.grid(True)
    plt.show()


"""Similarity matrix"""
def similarity_matrix(phonemes_neuralact, phonemes_chart, phoneme_clusteredchart):
    '''similarity matrix'''
    sp_flattened = [phonemes_neuralact[phonemes_chart[i]] for i in range(len(phonemes_chart))]

    sp_flattened -= np.mean(sp_flattened, axis = 0)

    ### generate dictionary 
    ### dictionary for phoneme's neural activities
    map_similarity = np.zeros((len(phonemes_chart), len(phonemes_chart)))

    ## generate shuffle map for the arrays according to phoneme clustering
    similarity_sp = np.zeros((len(phoneme_clusteredchart), len(phoneme_clusteredchart)))

    shuffle_sp_flattened = [sp_flattened[phonemes_chart.index(phone)] for phone in phoneme_clusteredchart]

    for i in range(len(phoneme_clusteredchart)):
        for j in range(i, len(phoneme_clusteredchart)):
    #         similarity_sp[i, j] = repre_similarity(shuffle_sp_flattened[i], shuffle_sp_flattened[j])
            similarity_sp[i, j] = loper.repre_similarity(shuffle_sp_flattened[i], shuffle_sp_flattened[j])
            similarity_sp[j, i] = similarity_sp[i, j]

    ## plot similarity
    plt.imshow(similarity_sp, cmap='RdBu')
    plt.title("Similarity: spike power")
    plt.colorbar()
    plt.xticks(ticks=np.arange(len(phoneme_clusteredchart)), labels=phoneme_clusteredchart, fontsize = 9, rotation=60)
    plt.yticks(ticks=np.arange(len(phoneme_clusteredchart)), labels=phoneme_clusteredchart, fontsize = 9)
    plt.show()

    plt.clf()


"""Confusion matrix"""
def confusion_matrix(feature, label, phoneme_clusteredchart, testsize, folds=10):
    '''confusion matrix'''
    ## ML route classify
    testsize = testsize

    cm_log = []
    y_test_log = []
    y_pred_log = []
    for _ in range(folds):
        X_train, X_test, y_train, y_test = train_test_split(feature,label, test_size = testsize, stratify = label)

        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        cm_log += [cm]
        y_test_log = np.append(y_test_log, y_test)
        y_pred_log = np.append(y_pred_log, y_pred)

    print(classification_report(y_test_log, y_pred_log))

    cm = np.array(cm_log).mean(0)
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.title(f"Confusion Matrix: {1 - testsize} train, sklearn classifier")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(ticks=np.arange(len(phoneme_clusteredchart)), labels=phoneme_clusteredchart, fontsize = 9, rotation = 60)
    plt.yticks(ticks=np.arange(len(phoneme_clusteredchart)), labels=phoneme_clusteredchart, fontsize = 9, rotation = 0)
    plt.show()
    plt.clf()


"""Learning curves"""
def learning_curves(results, outputpath, cases, title, xticks):
    # results: list of data; or data arrays.
    # cases: list of strings decribing the training and test set up for that case.
    if results.dtype.kind in {'U', 'S'}:
        num_cases = len(results)
        if num_cases != len(cases):
            raise Exception("Provide a list of all the cases of your examination. List should \
                            have a length equal to number of cases.")
        plt.figure(figsize=(8,6))
        for i in range(num_cases):
            result = pd.read_excel(results[i]).to_numpy()
            plt.errorbar(range(result.shape[1]), result[-2,:], result[-1,:], label=cases[i], marker='.')
        plt.title(title)
        plt.legend()
        plt.xticks(xticks[0], xticks[1])
        plt.savefig(outputpath)
        plt.show()
        plt.clf()

    if results.dtype.kind in {'i','u','f','c'}:
        num_cases = results.shape[0]
        if num_cases != len(cases):
            raise Exception("Provide a list of all the cases of your examination. List should \
                            have a length equal to number of cases.")
        plt.figure(figsize=(8,6))
        for i in range(num_cases):
            result = results[i, :, :]
            plt.errorbar(range(result.shape[1]), result[-2,:], result[-1,:], label=cases[i], marker='.')
        plt.title(title)
        plt.legend()
        plt.xticks(xticks[0], xticks[1])
        plt.savefig(outputpath)
        plt.show()
        plt.clf()
