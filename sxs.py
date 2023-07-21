import pandas as pd
#%matplotlib inline
import argparse
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument("new_data", help="New data to classify")
parser.add_argument("old_data", help="Old data that classify new data against")

args = parser.parse_args()

new_data=args.new_data
old_data = args.old_data

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

def classify_new_obs(fitted_classifier, new_data):
    '''
    Returns a prediction based on an amount of new data

    Inputs:
    fitted_classifier: fitted classification object
    new_data: new data of same type as previously trained data missing class

    '''
    new_data = pd.read_csv(new_data)
    print(fitted_classifier.predict(new_data.drop("subtype", axis=1)))

knn, x_test, y_test = train_classifier(old_data)
classify_new_obs(knn, new_data)