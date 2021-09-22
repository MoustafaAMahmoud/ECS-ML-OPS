#!/usr/bin/python3
# train.py

from sklearn import preprocessing
from joblib import dump
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import scipy
import numpy
import sys
import platform
import boto3
import time
import tempfile
print(platform.platform())
print("Python", sys.version)
print("NumPy", numpy.__version__)
print("SciPy", scipy.__version__)


def train():
    # Creating the high level object oriented interface
    client = boto3.client('s3')

    # Load, read and normalize training data
    bucket_name = 'sp-fargate-app-bucket'
    train_key_name = '/train.csv'

    # Read data from the S3 object
    data_train = pd.read_csv('s3://' + bucket_name + train_key_name)

    y_train = data_train['# Letter'].values
    X_train = data_train.drop(data_train.loc[:, 'Line':'# Letter'].columns, axis=1)

    # Data normalization (0,1)
    X_train = preprocessing.normalize(X_train, norm='l2')

    # Models training

    # Linear Discrimant Analysis (Default parameters)
    clf_lda = LinearDiscriminantAnalysis()
    clf_lda.fit(X_train, y_train)
    current_unix_time = str(int(time.time()))
    OutputFile_clf_lda = 'clf_lda_' + current_unix_time
    with tempfile.TemporaryFile() as fp:
        dump(clf_lda, fp)
        fp.seek(0)
        client.put_object(Bucket=bucket_name, Key=OutputFile_clf_lda, Body=fp.read())

    # Neural Networks multi-layer perceptron (MLP) algorithm
    clf_NN = MLPClassifier(solver='adam', activation='relu', alpha=0.0001, hidden_layer_sizes=(500,), random_state=0, max_iter=1000)
    clf_NN.fit(X_train, y_train)

    OutputFile_clf_NN = 'clf_NN_' + current_unix_time
    with tempfile.TemporaryFile() as fp:
        dump(clf_NN, fp)
        fp.seek(0)
        client.put_object(Bucket=bucket_name, Key=OutputFile_clf_NN, Body=fp.read())


if __name__ == '__main__':
    train()
