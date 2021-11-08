import numpy as np

import pandas as pd
import seaborn as sns

sns.set()

# ----------------------- PREPARATION DES DONNEES -----------------------

# Chargement des données
df = pd.read_csv('http://www.oasis-brains.org/pdf/oasis_longitudinal.csv')
# print("Premiers enregistrements:\n", df.head())

# Nettoyage du dataset
df = df.loc[df['Visit'] == 1]
# use first visit data only because of the analysis
df = df.reset_index(drop=True)
# reset index after filtering first visit data
df['M/F'] = df['M/F'].replace(['F', 'M'], [0, 1])  # M/F column
df['Group'] = df['Group'].replace(['Converted'], ['Demented'])  # Target variabl
df['Group'] = df['Group'].replace(['Demented', 'Nondemented'], [1, 0])  # Target
df = df.drop(['MRI ID', 'Visit', 'Hand'], axis=1)  # Drop unnecessary columns

# Check missing values by each column
pd.isnull(df).sum()
# The column, SES has 8 missing values

# Dropped the 8 rows with missing values in the column, SES
df_dropna = df.dropna(axis=0, how='any')
pd.isnull(df_dropna).sum()

df_dropna['Group'].value_counts()

# Draw scatter plot betweget_ipythonget_ipythonen EDUC and SES
x = df['EDUC']
y = df['SES']

ses_not_null_index = y[~y.isnull()].index
x = x[ses_not_null_index]
y = y[ses_not_null_index]

df.groupby(['EDUC'])['SES'].median()

df["SES"].fillna(df.groupby("EDUC")["SES"].transform("median"), inplace=True)

# I confirm there're no more missing values and all the 150 data were used.
pd.isnull(df['SES']).value_counts()

# Splitting Train/Validation/Test Sets

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score

# Dataset with imputation
Y = df['Group'].values  # Target for the model
X = df[['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']]  # Features we use

# splitting into three sets
X_trainval, X_test, Y_trainval, Y_test = train_test_split(X, Y, random_state=0)

# Feature scaling
scaler = MinMaxScaler().fit(X_trainval)
X_trainval_scaled = scaler.transform(X_trainval)
X_test_scaled = scaler.transform(X_test)

# Dataset after dropping missing value rows
Y = df_dropna['Group'].values  # Target for the model
X = df_dropna[['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']]  # Features we use

# splitting into three sets
X_trainval_dna, X_test_dna, Y_trainval_dna, Y_test_dna = train_test_split(X, Y, random_state=0)

# Feature scaling
scaler = MinMaxScaler().fit(X_trainval_dna)
X_trainval_scaled_dna = scaler.transform(X_trainval_dna)
X_test_scaled_dna = scaler.transform(X_test_dna)

# -----------------------  CONSTRUCTION DES MODELES -----------------------

# No convergence warnings
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

from sklearn.linear_model import LogisticRegression

import time

tps_tot = time.perf_counter()

acc = []  # list to store all performance metric

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import recall_score, roc_curve, auc

tps = time.perf_counter()

# Kernel Ridge imports
from sklearn.kernel_ridge import KernelRidge
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, roc_curve, auc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pylab import *
from sklearn.model_selection import cross_val_score


# Logistic Regression (with imputation)
def logistic_regression_imputation():
    tps = time.perf_counter()

    # Dataset with imputation

    best_score = 0
    kfolds = 5  # set the number of folds

    for c in [0.001, 0.1, 1, 10, 100]:
        logRegModel = LogisticRegression(C=c)
        # perform cross-validation

        # Get recall for each parameter setting
        scores = cross_val_score(logRegModel, X_trainval, Y_trainval, cv=kfolds, scoring='accuracy')

        # compute mean cross-validation accuracy
        score = np.mean(scores)

        # Find the best parameters and score
        if score > best_score:
            best_score = score
            best_parameters = c

    # rebuild a model on the combined training and validation set
    SelectedLogRegModel = LogisticRegression(C=best_parameters).fit(X_trainval_scaled, Y_trainval)

    test_score = SelectedLogRegModel.score(X_test_scaled, Y_test)
    PredictedOutput = SelectedLogRegModel.predict(X_test_scaled)
    test_recall = recall_score(Y_test, PredictedOutput, pos_label=1)
    fpr, tpr, thresholds = roc_curve(Y_test, PredictedOutput, pos_label=1)
    test_auc = auc(fpr, tpr)  # Calcul de l'air sous la courbe ROC
    print("Best accuracy on validation set is:", best_score)
    print("Best parameter for regularization (C) is: ", best_parameters)
    print("Test accuracy with best C parameter is", test_score)
    print("Test recall with the best C parameter is", test_recall)
    print("Test AUC with the best C parameter is", test_auc)
    """"
    print("Coefficient Logistic Regression", SelectedLogRegModel.coef_)
    plt.plot(fpr, tpr, color='red')
    plt.title('ROC')
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.show()
    """
    m = 'Logistic Regression (w/ imputation)'
    executionTime = "{:.2f}".format(time.perf_counter() - tps)
    acc.append([m, test_score, test_recall, test_auc, executionTime, fpr, tpr, thresholds])
    print("Execution time:", executionTime, "seconds")
    return "ok"


# Logistic Regression (with drop n/a)
def logistic_regression_dropNA():
    tps = time.perf_counter()

    # Dataset after dropping missing value rows
    best_parameters = 0
    best_score = 0
    kfolds = 5  # set the number of folds

    for c in [0.001, 0.1, 1, 10, 100]:
        logRegModel = LogisticRegression(C=c)
        # perform cross-validation
        scores = cross_val_score(logRegModel, X_trainval_scaled_dna, Y_trainval_dna, cv=kfolds, scoring='accuracy')

        # compute mean cross-validation accuracy
        score = np.mean(scores)

        # Find the best parameters and score
        if score > best_score:
            best_score = score
            best_parameters = c

    # rebuild a model on the combined training and validation set
    SelectedLogRegModel = LogisticRegression(C=best_parameters).fit(X_trainval_scaled_dna, Y_trainval_dna)

    test_score = SelectedLogRegModel.score(X_test_scaled_dna, Y_test_dna)
    PredictedOutput = SelectedLogRegModel.predict(X_test_scaled)
    test_recall = recall_score(Y_test, PredictedOutput, pos_label=1)
    fpr, tpr, thresholds = roc_curve(Y_test, PredictedOutput, pos_label=1)
    test_auc = auc(fpr, tpr)
    print("Best accuracy on validation set is:", best_score)
    print("Best parameter for regularization (C) is: ", best_parameters)
    print("Test accuracy with best C parameter is", test_score)
    print("Test recall with the best C parameter is", test_recall)
    print("Test AUC with the best C parameter is", test_auc)

    m = 'Logistic Regression (w/ dropna)'
    executionTime = "{:.2f}".format(time.perf_counter() - tps)
    acc.append([m, test_score, test_recall, test_auc, executionTime, fpr, tpr, thresholds])
    print("Execution time:", executionTime, "seconds")
    return {
            "best_score": best_score,
            "best_parameters": best_parameters,
            "test_score": test_score,
            "test_recall": test_recall,
            "test_auc": test_auc,
            "executionTime": executionTime
            }


# Analyse Discirminante Linéaire (LDA)
def analyse_linear_LDA():
    kfolds = 5  # set the number of folds
    best_score = 0

    logDiscriminantAnalysis = LinearDiscriminantAnalysis()
    # perform cross-validation
    scores = cross_val_score(logDiscriminantAnalysis, X_trainval_scaled_dna, Y_trainval_dna, cv=kfolds,
                             scoring='accuracy')
    # compute mean cross-validation accuracy
    score = np.mean(scores)

    # rebuild a model on the combined training and validation set
    SelectedDiscriminantAnalysisModel = LinearDiscriminantAnalysis().fit(X_trainval_scaled_dna, Y_trainval_dna)

    test_score_lda = SelectedDiscriminantAnalysisModel.score(X_test_scaled_dna, Y_test_dna)
    PredictedOutput_lda = SelectedDiscriminantAnalysisModel.predict(X_test_scaled)
    test_recall_lda = recall_score(Y_test, PredictedOutput_lda, pos_label=1)
    fpr_lda, tpr_lda, thresholds_lda = roc_curve(Y_test, PredictedOutput_lda, pos_label=1)
    test_auc_lda = auc(fpr_lda, tpr_lda)
    print("Test accuracy is", test_score_lda)
    print("Test recall is", test_recall_lda)
    print("Test AUC is", test_auc_lda)

    m = 'Linear Discriminant Analysis'
    executionTime = "{:.2f}".format(time.perf_counter() - tps)
    acc.append([m, test_score_lda, test_recall_lda, test_auc_lda, fpr_lda, tpr_lda, thresholds_lda])
    print("Execution time:", executionTime, "seconds")
    return {
        "test_score_lda": test_score_lda,
        "test_recall_lda": test_recall_lda,
        "test_auc_lda": test_auc_lda,
        "executionTime": executionTime
    }


# Analyse Discriminante Quadratique (QDA)
def analyse_linear_QDA():
    tps = time.perf_counter()
    kfolds = 5  # set the number of folds
    best_score = 0

    logQuadraticDiscriminantAnalysis = QuadraticDiscriminantAnalysis()
    # perform cross-validation
    scores = cross_val_score(logQuadraticDiscriminantAnalysis, X_trainval_scaled_dna, Y_trainval_dna, cv=kfolds,
                             scoring='accuracy')
    # compute mean cross-validation accuracy
    score = np.mean(scores)

    # rebuild a model on the combined training and validation set
    SelectedQuadraticDiscriminantAnalysisModel = QuadraticDiscriminantAnalysis().fit(X_trainval_scaled_dna,
                                                                                     Y_trainval_dna)

    test_score_qda = SelectedQuadraticDiscriminantAnalysisModel.score(X_test_scaled_dna, Y_test_dna)
    PredictedOutput_qda = SelectedQuadraticDiscriminantAnalysisModel.predict(X_test_scaled)
    test_recall_qda = recall_score(Y_test, PredictedOutput_qda, pos_label=1)
    fpr_qda, tpr_qda, thresholds_qda = roc_curve(Y_test, PredictedOutput_qda, pos_label=1)
    test_auc_qda = auc(fpr_qda, tpr_qda)
    print("Test accuracy is", test_score_qda)
    print("Test recall is", test_recall_qda)
    print("Test AUC is", test_auc_qda)

    m = 'Quadratic Discriminant Analysis'
    executionTime = "{:.2f}".format(time.perf_counter() - tps)
    acc.append([m, test_score_qda, test_recall_qda, test_auc_qda, fpr_qda, tpr_qda, thresholds_qda])
    print("Execution time:", executionTime, "seconds")
    return {
        "test_score_qda": test_score_qda,
        "test_recall_qda": test_recall_qda,
        "test_auc_qda": test_auc_qda,
        "executionTime": executionTime
    }


# Kernel Ridge
def kernel_ridge():
    pathToData = 'http://www.oasis-brains.org/pdf/oasis_longitudinal.csv'
    sns.set()

    df = pd.read_csv(pathToData)
    df = df.loc[df['Visit'] == 1]  # use first visit data only because of the analysis we're doing
    df = df.reset_index(drop=True)  # reset index after filtering first visit data
    df['M/F'] = df['M/F'].replace(['F', 'M'], [0, 1])  # M/F column
    df['Group'] = df['Group'].replace(['Converted'], ['Demented'])  # Target variable
    df['Group'] = df['Group'].replace(['Demented', 'Nondemented'], [1, 0])  # Target variable
    df = df.drop(['MRI ID', 'Visit', 'Hand'], axis=1)  # Drop unnecessary columns

    # The column, SES has 8 missing values

    # Dropped the 8 rows with missing values in the column, SES
    df_dropna = df.dropna(axis=0, how='any')
    pd.isnull(df_dropna).sum()
    df_dropna['M/F'].value_counts()

    # Dataset with imputation
    Y = df['Group'].values  # Target for the model
    X = df[['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']]  # Features we use

    # splitting into three sets
    X_trainval, X_test, Y_trainval, Y_test = train_test_split(X, Y, random_state=0)

    # Feature scaling
    scaler = MinMaxScaler().fit(X_trainval)
    X_trainval_scaled = scaler.transform(X_trainval)
    X_test_scaled = scaler.transform(X_test)

    # Dataset after dropping missing value rows
    Y = df_dropna['Group'].values  # Target for the model
    X = df_dropna[['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']]  # Features we use

    # splitting into three sets
    X_trainval_dna, X_test_dna, Y_trainval_dna, Y_test_dna = train_test_split(
        X, Y, random_state=0)

    # Feature scaling
    scaler = MinMaxScaler().fit(X_trainval_dna)
    X_trainval_scaled_dna = scaler.transform(X_trainval_dna)
    X_test_scaled_dna = scaler.transform(X_test_dna)

    clf = KernelRidge(alpha=1.0)
    clf.fit(X_trainval_scaled_dna, Y_trainval_dna)
    KernelRidge(alpha=1.0)
    clf.get_params(deep=True)
    PredictedOutput_qda = clf.predict(X_test_scaled_dna)
    score_krr = clf.score(X_test_scaled_dna, Y_test_dna, sample_weight=None)
    print("------- Kernel Ridge Regression ------- ")
    print("Test coefficient de détermination", score_krr)
    return {
        "score_krr": score_krr
    }
