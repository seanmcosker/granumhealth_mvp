#import protein_utilities as pu
import argparse


parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument("new_data", help="New data to classify")
parser.add_argument("old_data", help="Old data that classify new data against")

args = parser.parse_args()

new_data=args.new_data
old_data = args.old_data


####################################

import pandas as pd
import numpy as np
import regex as re
import random
import sklearn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def do_PCA(filename, filetype="csv", dim = 2):
    '''
    This is designed to fit a PCA on a file input. File should be a CSV or otherwise with no label column

    Inputs:
    filename: path to file to fit PCA on
    filetype: either CSV, table, excel (xls, etc)
    n_components: # of PCA dimensions to find

    Returns a fitted PCA dataframe
    '''
    if filetype == "csv":
        simul = pd.read_csv(filename)
    elif filetype == "table":
        simul = pd.read_table(filename)
    else:
        simul = pd.read_excel(filename)
    
    pca=PCA(dim)

    sdf = pca.fit_transform(simul)
    
    return sdf


def do_KMeans(df, clusters = 3):
    '''

    Perform KMeans on a PCA dataframe and visualize clusters

    Inputs:
    df: PCA-fitted DF
    clusters: number of PCA clusters. 3 recommended

    Outputs:
    
    Kmeans object
    '''

    kmeans = KMeans(clusters)
    kmeans.fit(df)
    y_kmeans=kmeans.predict(df)

   
    df = pd.DataFrame(df)
    #viz time
    
    df['subtype'] = y_kmeans

    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df['subtype'], s=50, cmap='viridis')

    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

    plt.savefig("clusters.png")


def train_classifier(data, filetype = "csv"):
    '''
    Helper function to train a classifier

    Inputs:
    data: data to classify
    filetype: type of data

    Outputs:
    Type of class
    '''
    if filetype == "csv":
        simul = pd.read_csv(data)
    elif filetype == "table":
        simul = pd.read_table(data)
    else:
        simul = pd.read_excel(data)

    print(type(simul))
    knn = KNeighborsClassifier()
    X_train, X_test, y_train, y_test = train_test_split(simul.drop("subtype", axis=1), simul['subtype'], random_state=42)
    knn.fit(X_train, y_train)
    return knn, X_test, y_test

def validate_classifier(classifier, X_test, y_test):

    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_pred, y_test)
    print(cm)

def classify_new_obs(fitted_classifier, new_data):
    '''
    Returns a prediction based on an amount of new data

    Inputs:
    fitted_classifier: fitted classification object
    new_data: new data of same type as previously trained data missing class

    '''
    new_data = pd.read_csv(new_data)
    print(fitted_classifier.predict(new_data.drop("subtype", axis=1)))



############ ACTUAL SCRIPT ################

#test_df = do_PCA(old_data)
knn, x_test, y_test = train_classifier(old_data)
classify_new_obs(knn, new_data)

