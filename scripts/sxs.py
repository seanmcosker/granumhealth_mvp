import pandas as pd
import argparse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument("new_data", help="New data to classify")
parser.add_argument("old_data", help="Old data that classify new data against")

args = parser.parse_args()

new_data=args.new_data
old_data = args.old_data

#print(type(new_data))

def read_data(data, filetype = "csv"):
    if filetype == "csv":
        df = pd.read_csv(data)
    elif filetype == "table":
        df = pd.read_table(data)
    else:
        df = pd.read_excel(data)
    
    return df


def format_responses(df):
    ''' 
    Formatting for responses
    '''
    cols_list = ["hash","username", "age", "gender", "lettuce-eat", "lettuce-enjoy", "cilantro-eat", "cilantro-enjoy", 'asp-eat', 'asp-enjoy', 'apple', 'pineapple', 'starfruit', 'fav-meal', 'fav-climate', 'startdate', 'submitdate', 'networkID', 'tags']
    #if len(cols_list) == df.shape[1]:
    df.columns = cols_list
    #print(df.columns)
    df.drop(["hash","username","startdate","submitdate","networkID","tags"], inplace=True, axis=1)


    change_cols = ["age", "gender", "apple", "pineapple", "starfruit", "fav-meal", "fav-climate"]
    df[change_cols] = df[change_cols].apply(lambda x: pd.factorize(x)[0] + 1)
    df = df.fillna(0)
    return df


def train_classifier(df):
    '''
    Helper function to train a classifier

    Inputs:
    data: data to classify
    filetype: type of data

    Outputs:
    Type of class
    '''

    #print(type(simul))
    knn = KNeighborsClassifier()
    X_train, X_test, y_train, y_test = train_test_split(df.drop("subtype", axis=1), df['subtype'], random_state=42)
    knn.fit(X_train, y_train)
    #print(X_train.columns)
    return knn, X_test, y_test

def classify_new_obs(fitted_classifier, new_data):
    '''
    Returns a prediction based on an amount of new data

    Inputs:
    fitted_classifier: fitted classification object
    new_data: new data of same type as previously trained data missing class

    '''
   #new_data = pd.read_csv(new_data)
    #print(new_data.columns)
    print( fitted_classifier.predict(new_data))

#old_data = format_responses(old_data)


old_data = read_data(old_data)
new_data = read_data(new_data)
new_data = format_responses(new_data)

##################

knn, X_test, y_test = train_classifier(old_data)
classify_new_obs(knn, new_data)
